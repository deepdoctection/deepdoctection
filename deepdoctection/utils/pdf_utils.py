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
Module with pdf processing tools
"""
import os
import platform
import subprocess
import sys
from errno import ENOENT
from io import BytesIO
from shutil import copyfile
from typing import Generator, List, Optional, Tuple

from numpy import uint8
from pypdf import PdfReader, PdfWriter, errors

from .context import save_tmp_file, timeout_manager
from .detection_types import ImageType, Pathlike
from .error import DependencyError, FileExtensionError
from .file_utils import pdf_to_cairo_available, pdf_to_ppm_available, qpdf_available
from .logger import LoggingRecord, logger
from .utils import is_file_extension
from .viz import viz_handler

__all__ = ["decrypt_pdf_document", "get_pdf_file_reader", "get_pdf_file_writer", "PDFStreamer", "pdf_to_np_array"]


def decrypt_pdf_document(path: Pathlike) -> bool:
    """
    Decrypting a pdf. As copying a pdf document removes the password that protects pdf, this method
    generates a copy and decrypts the copy using qpdf. The result is saved as the original
    document.

    qpdf: <http://qpdf.sourceforge.net/>

    Note, that this is decryption does not work, if the pdf has a readable protection, in which case we do not
    provide any solution.

    :param path: A path to the pdf file
    :return: True if document has been successfully decrypted
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


def get_pdf_file_reader(path: Pathlike) -> PdfReader:
    """
    Creates a file reader object from a pdf document. Will try to decrypt the document if it is
    encrypted. (See `decrypt_pdf_document` to understand what is meant with "decrypt").

    :param path: A path to a pdf document
    :return: A file reader object from which you can iterate through the document.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(str(path))
    file_name = os.path.split(path)[1]
    if not is_file_extension(file_name, ".pdf"):
        raise FileExtensionError(f"must be a pdf file: {file_name}")

    with open(path, "rb") as file:
        qpdf_called = False
        try:
            input_pdf_as_bytes = PdfReader(file)
        except (errors.PdfReadError, AttributeError):
            _ = decrypt_pdf_document(path)
            qpdf_called = True

        if not qpdf_called:
            if input_pdf_as_bytes.is_encrypted:
                is_decrypted = decrypt_pdf_document(path)
                if not is_decrypted:
                    logger.error(
                        LoggingRecord(
                            f"pdf document {path} cannot be decrypted and therefore cannot be " f"processed further."
                        )
                    )
                    sys.exit()

    file_reader = PdfReader(open(path, "rb"))  # pylint: disable=R1732
    return file_reader


def get_pdf_file_writer() -> PdfWriter:
    """
    `PdfWriter` instance
    """
    return PdfWriter()


class PDFStreamer:
    """
    A class for streaming pdf documents as bytes objects. Build as a generator, it is possible to load the document
    iteratively into memory. Uses py2pdf FileReader and FileWriter.

    **Example:**

             df = dataflow.DataFromIterable.PDFStreamer(path=path)
             df.reset_state()

             for page in df:
                ... # do whatever you like

    """

    def __init__(self, path: Pathlike) -> None:
        """
        :param path: to a pdf.
        """
        self.file_reader = get_pdf_file_reader(path)
        self.file_writer = PdfWriter()

    def __len__(self) -> int:
        return len(self.file_reader.pages)

    def __iter__(self) -> Generator[Tuple[bytes, int], None, None]:
        for k in range(len(self)):
            buffer = BytesIO()
            writer = get_pdf_file_writer()
            writer.add_page(self.file_reader.pages[k])
            writer.write(buffer)
            yield buffer.getvalue(), k


# The following functions are modified versions from the Python poppler wrapper
# https://github.com/Belval/pdf2image/blob/master/pdf2image/pdf2image.py


def _input_to_cli_str(
    input_file_name: Pathlike, output_file_name: Pathlike, dpi: int, size: Optional[Tuple[int, int]] = None
) -> List[str]:
    cmd_args: List[str] = []

    if pdf_to_ppm_available():
        command = "pdftoppm"
    elif pdf_to_cairo_available():
        command = "pdftocairo"
    else:
        raise DependencyError("Poppler not found. Please install or add to your PATH.")

    if platform.system() == "Windows":
        command = command + ".exe"
    cmd_args.append(command)
    cmd_args.extend(["-r", str(dpi), str(input_file_name)])
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
    Poppler Error
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__()
        self.status = status
        self.message = message
        self.args = (status, message)


def _run_poppler(poppler_args: List[str]) -> None:
    try:
        proc = subprocess.Popen(poppler_args)  # pylint: disable=R1732
    except OSError as error:
        if error.errno != ENOENT:
            raise error from error
        raise DependencyError("Poppler not found. Please install or add to your PATH.") from error

    with timeout_manager(proc, 0):
        if proc.returncode:
            raise PopplerError(status=proc.returncode, message="Syntax Error: PDF cannot be read with Poppler")


def pdf_to_np_array(pdf_bytes: bytes, size: Optional[Tuple[int, int]] = None, dpi: int = 200) -> ImageType:
    """
    Convert a single pdf page from its byte representation to a numpy array. This function will save the pdf as to a tmp
    file and then call poppler via `pdftoppm` resp. `pdftocairo` if the former is not available.

    :param pdf_bytes: Bytes representing the PDF file
    :param size: Size of the resulting image(s), uses (width, height) standard
    :param dpi:  Image quality in DPI/dots-per-inch (default 200)
    :return: numpy array
    """

    with save_tmp_file(pdf_bytes, "pdf_") as (tmp_name, input_file_name):
        _run_poppler(_input_to_cli_str(input_file_name, tmp_name, dpi, size))
        image = viz_handler.read_image(tmp_name + "-1.png")

    return image.astype(uint8)
