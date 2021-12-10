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
from shutil import which, copyfile
from typing import Generator, Tuple
from io import BytesIO

from PyPDF2 import PdfFileReader, PdfFileWriter  # type: ignore


def decrypt_pdf_document(path: str) -> None:
    """
    Decrypting a pdf. As copying a pdf document removes the password that protects pdf, this method
    generates a copy and decrypts the copy using qpdf. The result is saved as the original
    document.

    qpdf: http://qpdf.sourceforge.net/

    Note, that this is decryption does not work, if the pdf has a readable protection, in which case we do not
    provide any solution.

    :param path: A path to the pdf file
    """

    assert which("qpdf") is not None, "decrypt_pdf_document requires 'qpdf' to be installed."
    path_base, file_name = os.path.split(path)
    file_name_tmp = os.path.splitext(file_name)[0] + "tmp.pdf"
    path_tmp = os.path.join(path_base, file_name_tmp)
    copyfile(path, path_tmp)
    cmd_str = f"qpdf --password='' --decrypt {path_tmp} {os.path.join(path_base, path)}"
    os.system(cmd_str)
    os.remove(path_tmp)


def get_pdf_file_reader(path: str) -> PdfFileReader:
    """
    Creates a file reader object from a pdf document. Will try to decrypt the document if it is
    encrypted. (See :func:`decrypt_pdf_document` to understand what is meant with "decrypt").

    :param path: A path to a pdf document
    :return: A file reader object from which you can iterate through the document.
    """

    assert os.path.isfile(path), path
    file_name = os.path.split(path)[1]
    assert os.path.splitext(file_name)[-1].lower() == ".pdf", f"must be a pdf file: {file_name}"

    with open(path, "rb") as file:
        input_pdf_as_bytes = PdfFileReader(file)

        if input_pdf_as_bytes.isEncrypted:
            decrypt_pdf_document(path)

    file_reader = PdfFileReader(open(path, "rb"))  # pylint: disable=R1732
    return file_reader


def get_pdf_file_writer() -> PdfFileWriter:
    """
    PdfFileWriter instance
    """
    return PdfFileWriter()


class PDFStreamer:
    """
    A class for streaming pdf documents as bytes objects. Build as a generator, it is possible to load the document
    iteratively into memory. Uses py2pdf FileReader and FileWriter.

    Example:
         df = dataflow.DataFromIterable.PDFStreamer(path=path)
         df.reset_state()

         for page in df:
            ... # do whatever you like


    """

    def __init__(self, path: str) -> None:
        """
        :param path to a pdf.
        """
        self.file_reader = get_pdf_file_reader(path)
        self.file_writer = PdfFileWriter()

    def __len__(self) -> int:
        return self.file_reader.getNumPages()

    def __iter__(self) -> Generator[Tuple[bytes, int], None, None]:
        for k in range(len(self)):
            buffer = BytesIO()
            writer = PdfFileWriter()
            writer.addPage(self.file_reader.getPage(k))
            writer.write(buffer)
            yield buffer.getvalue(), k
