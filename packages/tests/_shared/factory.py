# -*- coding: utf-8 -*-
# File: factory.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass


_MINIMAL_LETTER_PDF_BYTES = (
    b"%PDF-1.3\n1 0 obj\n<<\n/Type /Pages\n/Count 1\n/Kids [ 3 0 R ]\n>>\nendobj\n"
    b"2 0 obj\n<<\n/Producer (PyPDF2)\n>>\nendobj\n"
    b"3 0 obj\n<<\n/Type /Page\n/Parent 1 0 R\n/Resources <<\n/Font <<\n/F1 5 0 R\n>>\n"
    b"/ProcSet 6 0 R\n>>\n/MediaBox [ 0 0 612 792 ]\n/Contents 7 0 R\n>>\nendobj\n"
    b"4 0 obj\n<<\n/Type /Catalog\n/Pages 1 0 R\n>>\nendobj\n"
    b"5 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/Name /F1\n/BaseFont /Helvetica\n/Encoding /WinAnsiEncoding\n>>\nendobj\n"
    b"6 0 obj\n[ /PDF /Text ]\nendobj\n"
    b"7 0 obj\n<<\n/Length 1074\n>>\nstream\n2 J\r\nBT\r\n0 0 0 rg\r\n/F1 0027 Tf\r\n57.3750 722.2800 Td\r\n( A Simple PDF File ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 688.6080 Td\r\n( This is a small demonstration .pdf file - ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 664.7040 Td\r\n( just for use in the Virtual Mechanics tutorials. More text. And more ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 652.7520 Td\r\n( text. And more text. And more text. And more text. ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 628.8480 Td\r\n( And more text. And more text. And more text. And more text. And more ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 616.8960 Td\r\n( text. And more text. Boring, zzzzz. And more text. And more text. And ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 604.9440 Td\r\n( more text. And more text. And more text. And more text. And more text. ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 592.9920 Td\r\n( And more text. And more text. ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 569.0880 Td\r\n( And more text. And more text. And more text. And more text. And more ) Tj\r\nET\r\n"
    b"BT\r\n/F1 0010 Tf\r\n69.2500 557.1360 Td\r\n( text. And more text. And more text. Even more. Continued on page 2 ...) Tj\r\nET\r\n\nendstream\nendobj\n"
    b"xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000068 00000 n \n0000000108 00000 n \n0000000251 00000 n \n0000000300 00000 n \n"
    b"0000000407 00000 n \n0000000437 00000 n \ntrailer\n<<\n/Size 8\n/Root 4 0 R\n/Info 2 0 R\n>>\nstartxref\n1563\n%%EOF\n"
)

@dataclass(frozen=True)
class TestPdfPage:
    """Immutable test helper for a 1-page Letter PDF in bytes."""
    pdf_bytes: bytes
    loc: str
    file_name: str
    np_array_shape_default: tuple[int, int, int] = (792, 612, 3)
    np_array_shape_300: tuple[int, int, int] = (3301, 2550, 3)

def build_test_pdf_page() -> TestPdfPage:
    """Return a deterministic 1-page PDF payload for tests."""
    return TestPdfPage(
        pdf_bytes=_MINIMAL_LETTER_PDF_BYTES,
        loc="/testlocation/test",
        file_name="test_image_0.pdf",
    )
