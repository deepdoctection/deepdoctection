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
import os
from base64 import b64encode
from io import BytesIO
from typing import Callable, List, Optional, Union
from urllib.request import urlretrieve

from cv2 import IMREAD_COLOR, imread

from ..utils.pdf_utils import get_pdf_file_reader, get_pdf_file_writer
from .detection_types import ImageType
from .logger import logger
from .tqdm import get_tqdm

__all__ = [
    "load_image_from_file",
    "load_bytes_from_pdf_file",
    "get_load_image_func",
    "maybe_path_or_pdf",
    "download",
    "mkdir_p",
]


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """
    Converting bytes number into human-readable
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")
def mkdir_p(dir_name: str) -> None:
    """
    Like "mkdir -p", make a dir recursively, but do nothing if the dir exists

    :param dir_name: name of dir
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
def download(url: str, directory: str, file_name: Optional[str] = None, expect_size: Optional[int] = None) -> str:
    """
    Download URL to a directory. Will figure out the filename automatically from URL, if not given.
    """
    mkdir_p(directory)
    if file_name is None:
        file_name = url.split("/")[-1]
    f_path = os.path.join(directory, file_name)

    if os.path.isfile(f_path):
        if expect_size is not None and os.stat(f_path).st_size == expect_size:
            logger.info("File %s exists! Skip download.", file_name)
            return f_path
        logger.warning("File %s exists. Will overwrite with a new download!", file_name)
    else:
        logger.info("File %s will be downloaded.", file_name)

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
            f_path, _ = urlretrieve(url, f_path, reporthook=hook(time))  # type: ignore
        stat_info = os.stat(f_path)
        size = stat_info.st_size
    except IOError:
        logger.error("Failed to download %s", url)
        raise
    assert size > 0, f"Downloaded an empty file from {url}!"

    if expect_size is not None and size != expect_size:
        logger.error("File downloaded from %s does not match the expected size!", url)
        logger.error("You may have downloaded a broken file, or the upstream may have modified the file.")

    logger.info("Successfully downloaded %s. %s.", file_name, sizeof_fmt(size))
    return f_path


def is_file_extension(file_name: str, extension: Union[str, List[str]]) -> bool:
    """
    Check if a given file name has a given extension

    :param file_name: the file name, either full along with path or as stand alone
    :param extension: the extension of the file. Must add a dot (.)
    :return: True/False
    """
    if isinstance(extension, str):
        return os.path.splitext(file_name)[-1].lower() == extension
    return os.path.splitext(file_name)[-1].lower() in extension


def load_image_from_file(path: str, type_id: str = "np") -> Optional[Union[str, ImageType]]:
    """
    Loads an image from path and passes back an encoded base64 string, a numpy array or None if file is not found
    or a conversion error occurs.

    :param path: A path to the image.
    :param type_id:  "np" or "b64".
    :return: image of desired representation
    """
    image: Optional[Union[str, ImageType]] = None

    assert is_file_extension(path, [".png", ".jpeg", ".jpg"]), f"image type not allowed: {path}"
    assert type_id in ("np", "b64"), "type not allowed"

    try:
        if type_id == "b64":
            with open(path, "rb") as file:
                image = b64encode(file.read()).decode("utf-8")
        else:
            image = imread(path, IMREAD_COLOR)
    except (FileNotFoundError, ValueError):
        pass

    return image


def load_bytes_from_pdf_file(path: str) -> bytes:
    """
    Loads a pdf file with one single page and passes back a bytes' representation of this file. Can be converted into
    a numpy or directly passed to the attr: image of Image.

    :param path: A path to a pdf file. If more pages are available, it will take the first page.
    :return: A bytes' representation of the file, width and height
    """

    assert is_file_extension(path, [".pdf"]), f"type not allowed: {path}"

    file_reader = get_pdf_file_reader(path)
    buffer = BytesIO()
    writer = get_pdf_file_writer()
    writer.addPage(file_reader.getPage(0))
    writer.write(buffer)
    return buffer.getvalue()


def get_load_image_func(
    path: str,
) -> Union[Callable[[str, str], Optional[Union[str, ImageType]]], Callable[[str], bytes]]:
    """
    Return the loading function according to its file extension.

    :param path: Path to a file
    :return: The function loading the file (and converting to its desired format)
    """

    assert is_file_extension(path, [".png", ".jpeg", ".jpg", ".pdf"]), f"image type not allowed: {path}"

    if is_file_extension(path, [".png", ".jpeg", ".jpg"]):
        return load_image_from_file
    if is_file_extension(path, [".pdf"]):
        return load_bytes_from_pdf_file
    return NotImplemented


def maybe_path_or_pdf(path: str) -> int:
    """
    Checks if the path points to a directory or a pdf document. Returns 1 if the path points to a directory, 2
    if the path points to a pdf doc or 0, if none of the previous is true.

    :param path: A path
    :return: A value of 0,1,2
    """

    is_dir = os.path.isdir(path)
    if is_dir:
        return 1
    file_name = os.path.split(path)[1]
    is_pdf = is_file_extension(file_name, ".pdf")
    if is_pdf:
        return 2
    return 0
