# -*- coding: utf-8 -*-
# File: fs.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Methods and classes that incorporate filesystem operations as well as file checks
"""

import errno
import json
import os
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from typing import Callable, Literal, Optional, Protocol, Union, overload
from urllib.request import urlretrieve

from .develop import deprecated
from .logger import LoggingRecord, logger
from .pdf_utils import get_pdf_file_reader, get_pdf_file_writer
from .settings import CACHE_DIR, CONFIGS, DATASET_DIR, MODEL_DIR, PATH
from .tqdm import get_tqdm
from .types import B64, B64Str, JsonDict, PathLikeOrStr, PixelValues
from .utils import is_file_extension
from .viz import viz_handler

__all__ = [
    "load_image_from_file",
    "load_bytes_from_pdf_file",
    "get_load_image_func",
    "maybe_path_or_pdf",
    "download",
    "mkdir_p",
    "load_json",
    "sub_path",
    "get_package_path",
    "get_cache_dir_path",
    "get_configs_dir_path",
    "get_weights_dir_path",
    "get_dataset_dir_path",
    "maybe_copy_config_to_cache",
]


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """
    Converts a number of bytes into a human-readable string.

    Example:
        ```python
        sizeof_fmt(1024)
        ```

    Args:
        num: The number of bytes.
        suffix: The suffix to use (default is `B`).

    Returns:
        A human-readable string representation of the byte size.
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")
def mkdir_p(dir_name: PathLikeOrStr) -> None:
    """
    Creates a directory recursively, similar to `mkdir -p`. Does nothing if the directory already exists.

    Example:
        ```python
        mkdir_p('/tmp/mydir')
        ```

    Args:
        dir_name: The name of the directory to create.

    Returns:
        None
    """
    assert dir_name is not None
    if dir_name == "" or os.path.isdir(dir_name):
        return None
    try:
        os.makedirs(dir_name)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise err


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")
def download(
    url: str, directory: PathLikeOrStr, file_name: Optional[str] = None, expect_size: Optional[int] = None
) -> str:
    """
    Downloads a file from a URL to a directory. Determines the filename from the URL if not provided.

    Example:
        ```python
        download('http://example.com/file.txt', '/tmp')
        ```

    Args:
        url: The URL to download from.
        directory: The directory to save the file in.
        file_name: The name of the file (optional).
        expect_size: The expected size of the file in bytes (optional).

    Returns:
        The path to the downloaded file.
    """
    mkdir_p(directory)
    if file_name is None:
        file_name = url.split("/")[-1]
    f_path = os.path.join(directory, file_name)

    if os.path.isfile(f_path):
        if (expect_size is not None and os.stat(f_path).st_size == expect_size) or expect_size is None:
            logger.info(LoggingRecord(f"File {file_name} exists! Skip download."))
            return f_path
        logger.warning(LoggingRecord(f"File {file_name} exists. Will overwrite with a new download!"))
    else:
        logger.info(LoggingRecord(f"File {file_name} will be downloaded."))

    def hook(total):  # type: ignore
        last_b = [0]

        def inner(byte, b_size, t_size=None):  # type: ignore
            if t_size is not None:
                total.total = t_size
            total.update((byte - last_b[0]) * b_size)
            last_b[0] = byte

        return inner

    try:
        with get_tqdm(unit="B", unit_scale=True, miniters=1, desc=file_name) as time:
            f_path, _ = urlretrieve(url, f_path, reporthook=hook(time))
        stat_info = os.stat(f_path)
        size = stat_info.st_size
    except IOError:
        logger.error(LoggingRecord(f"Failed to download {url}"))
        raise
    assert size > 0, f"Downloaded an empty file from {url}!"

    if expect_size is not None and size != expect_size:
        logger.warning(LoggingRecord(f"File downloaded from {url} does not match the expected size!"))
        logger.warning(
            LoggingRecord("You may have downloaded a broken file, or the upstream may have modified the file.")
        )

    logger.info(LoggingRecord(f"Successfully downloaded {file_name}. {sizeof_fmt(size)}."))
    return f_path


@overload
def load_image_from_file(path: PathLikeOrStr, type_id: Literal["np"] = "np") -> Optional[PixelValues]:
    ...


@overload
def load_image_from_file(path: PathLikeOrStr, type_id: Literal["b64"]) -> Optional[B64Str]:
    ...


def load_image_from_file(
    path: PathLikeOrStr, type_id: Literal["np", "b64"] = "np"
) -> Optional[Union[B64Str, PixelValues]]:
    """
    Loads an image from a file and returns either a base64-encoded string, a numpy array, or `None` if the file is not
    found or a conversion error occurs.

    Example:
        ```python
        load_image_from_file('image.png', type_id='b64')
        ```

    Args:
        path: The path to the image.
        type_id: The type of output, either `np` for numpy array or `b64` for base64 string.

    Returns:
        The image in the desired representation or `None`.
    """
    image: Optional[Union[str, PixelValues]] = None
    path = path.as_posix() if isinstance(path, Path) else path

    assert is_file_extension(path, [".png", ".jpeg", ".jpg", ".tif"]), f"image type not allowed: {path}"
    assert type_id in ("np", "b64"), "type not allowed"

    try:
        if type_id == "b64":
            with open(path, "rb") as file:
                image = b64encode(file.read()).decode("utf-8")
        else:
            image = viz_handler.read_image(path)
    except (FileNotFoundError, ValueError):
        logger.info(LoggingRecord(f"file not found or value error: {path}"))

    return image


def load_bytes_from_pdf_file(path: PathLikeOrStr, page_number: int = 0) -> B64:
    """
    Loads a PDF file with a single page and returns a bytes representation of this file. Can be converted into a numpy
    array or passed directly to the `image` attribute of `Image`.

    Example:
        ```python
        load_bytes_from_pdf_file('document.pdf', page_number=0)
        ```

    Args:
        path: The path to a PDF file. If more pages are available, it will take the first page.
        page_number: The page number to load. Raises `IndexError` if the document has fewer pages.

    Returns:
        A bytes representation of the file.

    """

    assert is_file_extension(path, [".pdf"]), f"type not allowed: {path}"

    file_reader = get_pdf_file_reader(path)
    buffer = BytesIO()
    writer = get_pdf_file_writer()
    writer.add_page(file_reader.pages[page_number])
    writer.write(buffer)
    return buffer.getvalue()


class LoadImageFunc(Protocol):
    """
    Protocol for typing `load_image_from_file`.

    Info:
        This protocol defines the call signature for image loading functions.
    """

    def __call__(self, path: PathLikeOrStr) -> Optional[PixelValues]:
        ...


def get_load_image_func(
    path: PathLikeOrStr,
) -> Union[LoadImageFunc, Callable[[PathLikeOrStr], B64]]:
    """
    Returns the loading function according to the file extension.

    Example:
        ```python
        get_load_image_func('image.png')
        ```

    Args:
        path: The path to a file.

    Returns:
        The function that loads the file and converts it to the desired format.

    Raises:
        NotImplementedError: If the file extension is not supported.
    """

    assert is_file_extension(path, [".png", ".jpeg", ".jpg", ".pdf", ".tif"]), f"image type not allowed: " f"{path}"

    if is_file_extension(path, [".png", ".jpeg", ".jpg", ".tif"]):
        return load_image_from_file
    if is_file_extension(path, [".pdf"]):
        return load_bytes_from_pdf_file
    raise NotImplementedError(
        "File extension not supported by any loader. Please specify a file type and raise an issue"
    )


def maybe_path_or_pdf(path: PathLikeOrStr) -> int:
    """
    Checks if the path points to a directory, a PDF document, or a single image.

    Example:
        ```python
        maybe_path_or_pdf('/path/to/file.pdf')
        ```

    Args:
        path: The path to check.

    Returns:
        `1` if the path points to a directory, `2` if it points to a PDF document, `3` if it points to a PNG, JPG,
            JPEG, or TIF image, or `0` otherwise.
    """

    if os.path.isdir(path):
        return 1
    file_name = os.path.split(path)[1]
    if is_file_extension(file_name, ".pdf"):
        return 2
    if is_file_extension(file_name, [".png", ".jpeg", ".jpg", ".tif"]):
        return 3
    return 0


def load_json(path_ann: PathLikeOrStr) -> JsonDict:
    """
    Loads a JSON file.

    Example:
        ```python
        load_json('annotations.json')
        ```

    Args:
        path_ann: The path to the JSON file.

    Returns:
        The loaded JSON as a dictionary.
    """
    with open(path_ann, "r", encoding="utf-8") as file:
        json_dict = json.loads(file.read())
    return json_dict


def get_package_path() -> Path:
    """
    Returns the full base path of this package.

    Returns:
        The base path of the package.
    """
    return PATH


def get_cache_dir_path() -> Path:
    """
    Returns the full base path to the cache directory.

    Returns:
        The base path to the cache directory.
    """
    return CACHE_DIR


def get_weights_dir_path() -> Path:
    """
    Returns the full base path to the model directory.

    Returns:
        The base path to the model directory.
    """
    return MODEL_DIR


def get_configs_dir_path() -> Path:
    """
    Returns the full base path to the configs directory.

    Returns:
        The base path to the configs directory.
    """
    return CONFIGS


def get_dataset_dir_path() -> Path:
    """
    Returns the full base path to the dataset directory.

    Returns:
        The base path to the dataset directory.
    """
    return DATASET_DIR


def maybe_copy_config_to_cache(
    package_path: PathLikeOrStr, configs_dir_path: PathLikeOrStr, file_name: str, force_copy: bool = True
) -> str:
    """
    Copies a file from the source directory to the target directory.

    Example:
        ```python
        maybe_copy_config_to_cache('/src', '/dst', 'config.yaml')
        ```

    Args:
        package_path: The base path to the source directory of the file.
        configs_dir_path: The base path to the target directory.
        file_name: The name of the file to copy.
        force_copy: If `True`, will re-copy the file even if it already exists in the target directory.

    Returns:
        The path to the copied file.
    """

    absolute_path_source = os.path.join(package_path, file_name)
    absolute_path = os.path.join(configs_dir_path, os.path.join(os.path.split(file_name)[1]))
    mkdir_p(os.path.split(absolute_path)[0])
    if not os.path.isfile(absolute_path) or force_copy:
        copyfile(absolute_path_source, absolute_path)
    return absolute_path


@deprecated("Use pathlib operations instead", "2022-06-08")
def sub_path(anchor_dir: PathLikeOrStr, *paths: PathLikeOrStr) -> PathLikeOrStr:
    """
    Generates a path from the anchor directory and additional path arguments.

    Example:
        ```python
        sub_path('/path/to', 'dir1', 'dir2')
        ```

    Args:
        anchor_dir: The anchor directory.
        *paths: Additional directories to add to the path.

    Returns:
        The generated sub-path.

    Note:
        Deprecated. Use pathlib operations instead.
    """
    return os.path.join(os.path.dirname(os.path.abspath(anchor_dir)), *paths)
