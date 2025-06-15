# -*- coding: utf-8 -*-
# File: pdf_utils.py

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
Pdf processing tools
"""

import os
import platform
import subprocess
import sys
from errno import ENOENT
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from typing import Generator, Literal, Optional, Union

from lazy_imports import try_import
from numpy import uint8
from pypdf import PdfReader, PdfWriter, errors

from .context import save_tmp_file, timeout_manager
from .env_info import ENV_VARS_TRUE
from .error import DependencyError, FileExtensionError
from .file_utils import pdf_to_cairo_available, pdf_to_ppm_available, qpdf_available
from .logger import LoggingRecord, logger
from .types import PathLikeOrStr, PixelValues
from .utils import is_file_extension
from .viz import viz_handler

with try_import() as pt_import_guard:
    import pypdfium2

__all__ = [
    "decrypt_pdf_document",
    "decrypt_pdf_document_from_bytes",
    "get_pdf_file_reader",
    "get_pdf_file_writer",
    "PDFStreamer",
    "pdf_to_np_array",
    "split_pdf",
]


def decrypt_pdf_document(path: PathLikeOrStr) -> bool:
    """
    Decrypt a PDF file.

    As copying a PDF document removes the password that protects the PDF, this method generates a copy and decrypts the
    copy using `qpdf`. The result is saved as the original document.

    Note:
        This decryption does not work if the PDF has a readable protection, in which case no solution is provided.
        `qpdf`: <http://qpdf.sourceforge.net/>

    Args:
        path: A path to the PDF file.

    Returns:
        True if the document has been successfully decrypted.
    """
    if qpdf_available():
        path_base, file_name = os.path.split(path)
        file_name_tmp = os.path.splitext(file_name)[0] + "tmp.pdf"
        path_tmp = os.path.join(path_base, file_name_tmp)
        copyfile(path, path_tmp)
        cmd_str = f"qpdf --password='' --decrypt {path_tmp} {os.path.join(path_base, path)}"
        response = os.system(cmd_str)
        os.remove(path_tmp)
        if not response:
            return True
    else:
        logger.info(
            LoggingRecord("qpdf is not installed. If the document must be decrypted please ensure that it is installed")
        )
    return False


def decrypt_pdf_document_from_bytes(input_bytes: bytes) -> bytes:
    """
    Decrypt a PDF given as bytes.

    Under the hood, it saves the bytes to a temporary file and then calls `decrypt_pdf_document`.

    Note:
        `qpdf`: <http://qpdf.sourceforge.net/>

    Args:
        input_bytes: A bytes object representing the PDF file.

    Returns:
        The decrypted bytes object.


    """
    with save_tmp_file(input_bytes, "pdf_") as (_, input_file_name):
        is_decrypted = decrypt_pdf_document(input_file_name)
        if is_decrypted:
            with open(input_file_name, "rb") as file:
                return file.read()
        else:
            logger.error(LoggingRecord("pdf bytes cannot be decrypted and therefore cannot be processed further."))
            sys.exit()


def get_pdf_file_reader(path_or_bytes: Union[PathLikeOrStr, bytes]) -> PdfReader:
    """
    Create a file reader object from a PDF document.

    Will try to decrypt the document if it is encrypted. (See `decrypt_pdf_document` to understand what is meant with
    "decrypt").

    Args:
        path_or_bytes: A path to a PDF document or bytes.

    Returns:
        A file reader object from which you can iterate through the document.
    """

    if isinstance(path_or_bytes, bytes):
        try:
            reader = PdfReader(BytesIO(path_or_bytes))
        except (errors.PdfReadError, AttributeError):
            decrypted_bytes = decrypt_pdf_document_from_bytes(path_or_bytes)
            reader = PdfReader(BytesIO(decrypted_bytes))
        return reader

    if not os.path.isfile(path_or_bytes):
        raise FileNotFoundError(str(path_or_bytes))
    file_name = os.path.split(path_or_bytes)[1]
    if not is_file_extension(file_name, ".pdf"):
        raise FileExtensionError(f"must be a pdf file: {file_name}")

    with open(path_or_bytes, "rb") as file:
        qpdf_called = False
        try:
            reader = PdfReader(file)
        except (errors.PdfReadError, AttributeError):
            _ = decrypt_pdf_document(path_or_bytes)
            qpdf_called = True

        if not qpdf_called:
            if reader.is_encrypted:
                is_decrypted = decrypt_pdf_document(path_or_bytes)
                if not is_decrypted:
                    logger.error(
                        LoggingRecord(
                            f"pdf document {path_or_bytes} cannot be decrypted and therefore cannot "
                            f"be processed further."
                        )
                    )
                    sys.exit()

    return PdfReader(os.fspath(path_or_bytes))


def get_pdf_file_writer() -> PdfWriter:
    """
    `PdfWriter` instance.

    Returns:
        A new `PdfWriter` instance.
    """
    return PdfWriter()


class PDFStreamer:
    """
    A class for streaming PDF documents as bytes objects.

    Built as a generator, it is possible to load the document iteratively into memory. Uses `pypdf` `PdfReader` and
    `PdfWriter`.

    Example:
        ```python
        df = dataflow.DataFromIterable(PDFStreamer(path=path))
        df.reset_state()
        for page in df:
            ...
        streamer = PDFStreamer(path=path)
        pages = len(streamer)
        random_int = random.sample(range(0, pages), 2)
        for ran in random_int:
            pdf_bytes = streamer[ran]
        streamer.close()
        ```

    Note:
        Do not forget to close the streamer, otherwise the file will never be closed and might cause memory leaks if
        you open many files.
    """

    def __init__(self, path_or_bytes: Union[PathLikeOrStr, bytes]) -> None:
        """
        Args:
            path_or_bytes: Path to a PDF.

        Returns:
            None.
        """
        self.file_reader = get_pdf_file_reader(path_or_bytes)
        self.file_writer = PdfWriter()

    def __len__(self) -> int:
        return len(self.file_reader.pages)

    def __iter__(self) -> Generator[tuple[bytes, int], None, None]:
        for k in range(len(self)):
            buffer = BytesIO()
            writer = get_pdf_file_writer()
            writer.add_page(self.file_reader.pages[k])
            writer.write(buffer)
            yield buffer.getvalue(), k
        self.file_reader.close()

    def __getitem__(self, index: int) -> bytes:
        buffer = BytesIO()
        writer = get_pdf_file_writer()
        writer.add_page(self.file_reader.pages[index])
        writer.write(buffer)
        return buffer.getvalue()

    def close(self) -> None:
        """
        Close the file reader
        """
        self.file_reader.close()


# The following functions are modified versions from the Python poppler wrapper
# https://github.com/Belval/pdf2image/blob/master/pdf2image/pdf2image.py


def _input_to_cli_str(
    input_file_name: PathLikeOrStr,
    output_file_name: PathLikeOrStr,
    dpi: Optional[int] = None,
    size: Optional[tuple[int, int]] = None,
) -> list[str]:
    cmd_args: list[str] = []

    if pdf_to_ppm_available():
        command = "pdftoppm"
    elif pdf_to_cairo_available():
        command = "pdftocairo"
    else:
        raise DependencyError("Poppler not found. Please install or add to your PATH.")

    if platform.system() == "Windows":
        command = command + ".exe"
    cmd_args.append(command)

    if dpi:
        cmd_args.extend(["-r", str(dpi)])
    cmd_args.append(str(input_file_name))
    cmd_args.append("-png")
    cmd_args.append(str(output_file_name))

    if size:
        assert len(size) == 2, size
        assert isinstance(size[0], int) and isinstance(size[1], int), size
        cmd_args.extend(["-scale-to-x", str(size[0])])
        cmd_args.extend(["-scale-to-y", str(size[1])])

    return cmd_args


class PopplerError(RuntimeError):
    """
    Poppler Error.
    """

    def __init__(self, status: int, message: str) -> None:
        """
        Args:
            status: Error status code.
            message: Error message.
        """
        super().__init__()
        self.status = status
        self.message = message
        self.args = (status, message)


def _run_poppler(poppler_args: list[str]) -> None:
    try:
        proc = subprocess.Popen(poppler_args)  # pylint: disable=R1732
    except OSError as error:
        if error.errno != ENOENT:
            raise error from error
        raise DependencyError("Poppler not found. Please install or add to your PATH.") from error

    with timeout_manager(proc, 0):
        if proc.returncode:
            raise PopplerError(status=proc.returncode, message="Syntax Error: PDF cannot be read with Poppler")


def pdf_to_np_array_poppler(
    pdf_bytes: bytes, size: Optional[tuple[int, int]] = None, dpi: Optional[int] = None
) -> PixelValues:
    """
    Convert a single PDF page from its byte representation to a numpy array using Poppler.

    This function will save the PDF as a temporary file and then call Poppler via `pdftoppm` or `pdftocairo`.

    Raises:
        ValueError: If neither `dpi` nor `size` is provided.

    Args:
        pdf_bytes: Bytes representing the PDF file.
        size: Size of the resulting image(s), as (width, height).
        dpi: Image quality in DPI/dots-per-inch.

    Returns:
        `np.array`.
    """
    if dpi is None and size is None:
        raise ValueError("Either dpi or size must be provided.")
    with save_tmp_file(pdf_bytes, "pdf_") as (tmp_name, input_file_name):
        _run_poppler(_input_to_cli_str(input_file_name, tmp_name, dpi, size))
        image = viz_handler.read_image(tmp_name + "-1.png")

    return image.astype(uint8)


def pdf_to_np_array_pdfmium(pdf_bytes: bytes, dpi: Optional[int] = None) -> PixelValues:
    """
    Convert a single PDF page from its byte representation to a numpy array using pdfium.

    Args:
        pdf_bytes: Bytes representing the PDF file.
        dpi: Image quality in DPI/dots-per-inch.

    Returns:
        `np.array`.

    Raises:
        ValueError: If `dpi` is not provided.
    """
    if dpi is None:
        raise ValueError("dpi must be provided.")
    page = pypdfium2.PdfDocument(pdf_bytes)[0]
    return page.render(scale=dpi * 1 / 72).to_numpy().astype(uint8)


def pdf_to_np_array(pdf_bytes: bytes, size: Optional[tuple[int, int]] = None, dpi: Optional[int] = None) -> PixelValues:
    """
    Convert a single PDF page from its byte representation to a `np.array`.

    This function will either use Poppler or pdfium to render the PDF.

    Args:
        pdf_bytes: Bytes representing the PDF file.
        size: Size of the resulting image(s), as (width, height).
        dpi: Image quality in DPI/dots-per-inch.

    Returns:
        `np.array`.

    Note:
        If `USE_DD_PDFIUM` is set, `pdf_to_np_array_pdfmium` does not support the `size` parameter and will use
        `dpi` instead.
    """
    if os.environ.get("USE_DD_PDFIUM", "False") in ENV_VARS_TRUE:
        if size is not None:
            logger.warning(
                LoggingRecord(
                    f"pdf_to_np_array_pdfmium does not support the size parameter. Will use dpi = {dpi} instead."
                )
            )
        return pdf_to_np_array_pdfmium(pdf_bytes, dpi)
    return pdf_to_np_array_poppler(pdf_bytes, size, dpi)


def split_pdf(
    pdf_path: PathLikeOrStr, output_dir: PathLikeOrStr, file_type: Literal["image", "pdf"], dpi: int = 200
) -> None:
    """
    Split a PDF into single pages.

    The pages are saved as single PDF or PNG files in a subfolder of the output directory.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Path to the output directory.
        file_type: Type of the output file. Either "image" or "pdf".
        dpi: Image quality in DPI/dots-per-inch.

    Returns:
        None.
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.stem
    output_dir = Path(output_dir)
    file_dir = output_dir / filename
    if not file_dir.exists():
        os.makedirs(file_dir)

    with open(pdf_path, "rb") as file:
        pdf = PdfReader(file)
        for i, page in enumerate(pdf.pages):
            writer = PdfWriter()
            writer.add_page(page)
            if file_type == ".pdf":
                with open(file_dir / f"{filename}_{i}.pdf", "wb") as out:
                    writer.write(out)
                    writer.close()
            else:
                with BytesIO() as buffer:
                    writer.write(buffer)
                    buffer.seek(0)
                    np_image = pdf_to_np_array(buffer.getvalue(), dpi=dpi)
                    viz_handler.write_image(file_dir / f"{filename}_{i}.png", np_image)
                    writer.close()
