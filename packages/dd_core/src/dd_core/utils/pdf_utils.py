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

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from errno import ENOENT
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal, Optional, Union

from lazy_imports import try_import
from numpy import uint8

from .context import save_tmp_file, timeout_manager
from .env_info import ENV_VARS_TRUE
from .error import DependencyError, FileExtensionError, PopplerError
from .file_utils import pdf_to_cairo_available, pdf_to_ppm_available, pikepdf_available, pypdf_available
from .logger import LoggingRecord, logger
from .types import B64, PathLikeOrStr, PixelValues
from .utils import is_file_extension
from .viz import viz_handler

with try_import() as pypdf_import_guard:
    from pypdf import PdfReader, PdfWriter
    from pypdf import errors as pypdf_errors


with try_import() as pt_import_guard:
    import pypdfium2

with try_import() as pikepdf_import_guard:
    import pikepdf

__all__ = [
    "PdfDecryptStatus",
    "PdfDecryptReport",
    "inspect_and_maybe_decrypt_pdf_bytes",
    "inspect_and_maybe_decrypt_pdf_file",
    "decrypt_pdf_document",
    "decrypt_pdf_document_from_bytes",
    "get_pdf_file_reader",
    "get_pdf_file_writer",
    "PDFStreamer",
    "pdf_to_np_array",
    "split_pdf",
    "load_bytes_from_pdf_file",
    "PopplerError",
    "pdf_to_np_array_poppler",
    "pdf_to_np_array_pdfmium",
]


class PdfDecryptStatus(str, Enum):
    """
    Status values for PDF inspection and decryption.
    """

    NOT_ENCRYPTED = "not_encrypted"
    DECRYPTED = "decrypted"
    INVALID_PDF = "invalid_pdf"


@dataclass(slots=True)
class PdfDecryptReport:
    """Report returned by PDF inspection and decryption helpers.

    Attributes:
        status: Final processing status.
        was_encrypted: Whether the input PDF was encrypted.
        syntax_issues: Syntax problems reported by pikepdf.
        parser_warnings: Parser warnings reported by pikepdf.
        encryption_bits: Encryption bit size if available.
        encryption_revision: Encryption revision if available.
        encryption_version: Encryption version if available.
        message: Optional human-readable status message.
    """

    status: PdfDecryptStatus
    was_encrypted: bool = False
    syntax_issues: list[str] = field(default_factory=list)
    parser_warnings: list[str] = field(default_factory=list)
    encryption_bits: int | None = None
    encryption_revision: int | None = None
    encryption_version: int | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether the PDF is processable after inspection."""

        return self.status in {PdfDecryptStatus.NOT_ENCRYPTED, PdfDecryptStatus.DECRYPTED}


def _collect_report(pdf: pikepdf.Pdf, syntax_check: bool, strict_syntax: bool) -> PdfDecryptReport:
    """Collect inspection metadata for an opened PDF.

    Args:
        pdf: Open pikepdf document.
        syntax_check: Whether to collect syntax issues.
        strict_syntax: Whether syntax issues should mark the PDF as invalid.

    Returns:
        A populated decryption report.
    """

    parser_warnings = [str(warning) for warning in pdf.get_warnings()]
    syntax_issues = [str(warning) for warning in pdf.check_pdf_syntax()] if syntax_check else []
    was_encrypted = bool(pdf.is_encrypted)
    encryption_bits = None
    encryption_revision = None
    encryption_version = None

    if was_encrypted:
        try:
            encryption_bits = int(pdf.encryption.bits)
            encryption_revision = int(pdf.encryption.R)
            encryption_version = int(pdf.encryption.V)
        except (AttributeError, TypeError, ValueError):
            pass

    if strict_syntax and syntax_issues:
        return PdfDecryptReport(
            status=PdfDecryptStatus.INVALID_PDF,
            was_encrypted=was_encrypted,
            syntax_issues=syntax_issues,
            parser_warnings=parser_warnings,
            encryption_bits=encryption_bits,
            encryption_revision=encryption_revision,
            encryption_version=encryption_version,
            message="PDF has syntax issues and strict_syntax=True.",
        )

    return PdfDecryptReport(
        status=PdfDecryptStatus.DECRYPTED if was_encrypted else PdfDecryptStatus.NOT_ENCRYPTED,
        was_encrypted=was_encrypted,
        syntax_issues=syntax_issues,
        parser_warnings=parser_warnings,
        encryption_bits=encryption_bits,
        encryption_revision=encryption_revision,
        encryption_version=encryption_version,
    )


def inspect_and_maybe_decrypt_pdf_bytes(
    input_bytes: bytes,
    syntax_check: bool = True,
    strict_syntax: bool = False,
    attempt_recovery: bool = True,
) -> tuple[PdfDecryptReport, bytes]:
    """
    Inspect PDF bytes and remove encryption when possible.

    Args:
        input_bytes: PDF bytes to inspect.
        syntax_check: Whether to collect syntax issues via pikepdf.
        strict_syntax: Whether syntax issues should raise `ValueError`.
        attempt_recovery: Whether pikepdf should attempt PDF recovery.

    Returns:
        A tuple containing the inspection report and the resulting PDF bytes.

    Raises:
        ImportError: If pikepdf is not installed.
        ValueError: If `strict_syntax` is enabled and syntax issues are found.
        pikepdf.PasswordError: If the PDF requires a password.
        pikepdf.PdfError: If the PDF cannot be parsed.
    """

    if not pikepdf_available():
        raise ImportError("pikepdf is not installed.")

    with pikepdf.Pdf.open(
        BytesIO(input_bytes),
        password="",
        suppress_warnings=True,
        attempt_recovery=attempt_recovery,
    ) as pdf:
        report = _collect_report(pdf, syntax_check, strict_syntax)

        if report.status == PdfDecryptStatus.INVALID_PDF:
            raise ValueError(report.message or "PDF has syntax issues and strict_syntax=True.")

        if report.status == PdfDecryptStatus.NOT_ENCRYPTED:
            return report, input_bytes

        output_buffer = BytesIO()
        pdf.save(output_buffer)
        return report, output_buffer.getvalue()



def inspect_and_maybe_decrypt_pdf_file(
    path: PathLikeOrStr,
    syntax_check: bool = True,
    strict_syntax: bool = False,
    attempt_recovery: bool = True,
    overwrite_input: bool = True,
) -> PdfDecryptReport:
    """
    Inspect a PDF file and remove encryption in place when possible.

    Args:
        path: Path to the PDF file.
        syntax_check: Whether to collect syntax issues via pikepdf.
        strict_syntax: Whether syntax issues should raise `ValueError`.
        attempt_recovery: Whether pikepdf should attempt PDF recovery.
        overwrite_input: Whether pikepdf may overwrite the input file while saving.

    Returns:
        The inspection report.

    Raises:
        ImportError: If pikepdf is not installed.
        FileNotFoundError: If the path does not exist.
        ValueError: If `strict_syntax` is enabled and syntax issues are found or if `path` is not a file.
        pikepdf.PasswordError: If the PDF requires a password.
        pikepdf.PdfError: If the PDF cannot be parsed.
    """

    if not pikepdf_available():
        raise ImportError("pikepdf is not installed.")

    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    with pikepdf.Pdf.open(
        pdf_path,
        password="",
        suppress_warnings=True,
        attempt_recovery=attempt_recovery,
        allow_overwriting_input=overwrite_input,
    ) as pdf:
        report = _collect_report(pdf, syntax_check, strict_syntax)

        if report.status == PdfDecryptStatus.INVALID_PDF:
            raise ValueError(report.message or "PDF has syntax issues and strict_syntax=True.")

        if report.status == PdfDecryptStatus.NOT_ENCRYPTED:
            return report

        pdf.save(pdf_path)
        return report



def decrypt_pdf_document(path: PathLikeOrStr) -> bool:
    """
    Decrypt a PDF file in place when possible.

    Args:
        path: Path to the PDF file.

    Returns:
        True if the PDF is already processable or was decrypted successfully, otherwise False.
    """

    if not pikepdf_available():
        logger.error(
            LoggingRecord("pikepdf is not installed. If the document must be decrypted please ensure that it is installed")
        )
        return False

    try:
        report = inspect_and_maybe_decrypt_pdf_file(path, True, False, True, True)
    except (FileNotFoundError, ValueError, pikepdf.PasswordError, pikepdf.PdfError) as exc:
        logger.error(LoggingRecord(f"PDF could not be decrypted or validated: {path} ({exc})"))
        return False

    if report.parser_warnings:
        logger.warning(LoggingRecord(f"PDF parser warnings for {path}: {report.parser_warnings[:5]}"))
    if report.syntax_issues:
        logger.warning(LoggingRecord(f"PDF syntax issues for {path}: {report.syntax_issues[:5]}"))

    return True


def decrypt_pdf_document_from_bytes(input_bytes: bytes) -> bytes:
    """
    Decrypt PDF bytes when possible.

    Args:
        input_bytes: PDF bytes to inspect and optionally decrypt.

    Returns:
        The original bytes for unencrypted PDFs or decrypted bytes for encrypted PDFs.

    Raises:
        ImportError: If pikepdf is not installed.
        ValueError: If `strict_syntax` is enabled and syntax issues are found.
        pikepdf.PasswordError: If the PDF requires a password.
        pikepdf.PdfError: If the PDF cannot be parsed.
    """

    report, output_bytes = inspect_and_maybe_decrypt_pdf_bytes(input_bytes, True, False, True)

    if report.parser_warnings:
        logger.warning(LoggingRecord(f"PDF parser warnings for input bytes: {report.parser_warnings[:5]}"))
    if report.syntax_issues:
        logger.warning(LoggingRecord(f"PDF syntax issues for input bytes: {report.syntax_issues[:5]}"))

    return output_bytes


def get_pdf_file_reader(path_or_bytes: Union[PathLikeOrStr, bytes], check_file_extension: bool = True) -> PdfReader:
    """
    Create a file reader object from a PDF document.

    Will try to decrypt the document if it is encrypted. (See `decrypt_pdf_document` to understand what is meant with
    "decrypt").

    Args:
        path_or_bytes: A path to a PDF document or bytes.
        check_file_extension: If True, and file suffix is not .pdf, it will raise a FileExtensionError

    Returns:
        A file reader object from which you can iterate through the document.
    """
    if not pypdf_available():
        raise ImportError("pypdf is not installed.")

    if isinstance(path_or_bytes, bytes):
        try:
            reader = PdfReader(BytesIO(path_or_bytes))
        except (pypdf_errors.PdfReadError, AttributeError):
            decrypted_bytes = decrypt_pdf_document_from_bytes(path_or_bytes)
            reader = PdfReader(BytesIO(decrypted_bytes))
        return reader

    if not os.path.isfile(path_or_bytes):
        raise FileNotFoundError(str(path_or_bytes))
    file_name = os.path.split(path_or_bytes)[1]
    if not is_file_extension(file_name, ".pdf") and check_file_extension:
        raise FileExtensionError(f"must be a pdf file: {file_name}")

    with open(path_or_bytes, "rb") as file:
        qpdf_called = False
        try:
            reader = PdfReader(file)
        except (pypdf_errors.PdfReadError, AttributeError):
            is_decrypted = decrypt_pdf_document(path_or_bytes)
            qpdf_called = True
            if not is_decrypted:
                logger.error(
                    LoggingRecord(
                        f"pdf document {path_or_bytes} cannot be decrypted and therefore cannot be processed further."
                    )
                )
                sys.exit()

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
    if not pypdf_available():
        raise ImportError("pypdf is not installed.")
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

    def __init__(self, path_or_bytes: Union[PathLikeOrStr, bytes], check_file_extension: bool = True) -> None:
        """
        Args:
            path_or_bytes: Path to a PDF.
            check_file_extension: If True, and file suffix is not .pdf, it will raise a FileExtensionError


        Returns:
            None.
        """
        self.file_reader = get_pdf_file_reader(path_or_bytes, check_file_extension=check_file_extension)
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
    if os.environ["USE_DD_PDFIUM"] in ENV_VARS_TRUE:
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
