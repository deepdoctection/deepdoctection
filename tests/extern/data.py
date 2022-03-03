# -*- coding: utf-8 -*-
# File: data.py

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
Some data samples in a separate module
"""

PDF_BYTES = (
    b"%PDF-1.3\n1 0 obj\n<<\n/Type /Pages\n/Count 1\n/Kids [ 3 0 R ]\n>>\nendobj\n2 0 obj\n<<\n/Producer "
    b"(PyPDF2)\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 1 0 R\n/Resources <<\n/Font "
    b"<<\n/F1 5 0 R\n>>\n/ProcSet 6 0 R\n>>\n/MediaBox [ 0 0 612 792 ]\n/Contents 7 0 R\n>>\nendobj\n4 "
    b"0 obj\n<<\n/Type /Catalog\n/Pages 1 0 R\n>>\nendobj\n5 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/Name"
    b" /F1\n/BaseFont /Helvetica\n/Encoding /WinAnsiEncoding\n>>\nendobj\n6 0 obj\n[ /PDF /Text"
    b" ]\nendobj\n7 0 obj\n<<\n/Length 1074\n>>\nstream\n2 J\r\nBT\r\n0 0 0 rg\r\n/F1 0027 Tf\r\n57.3750"
    b" 722.2800 Td\r\n( A Simple PDF File ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 688.6080 Td\r\n("
    b" This is a small demonstration .pdf file - ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 664.7040"
    b" Td\r\n( just for use in the Virtual Mechanics tutorials. More text. And more ) Tj\r\nET\r\nBT\r\n/F1"
    b" 0010 Tf\r\n69.2500 652.7520 Td\r\n( text. And more text. And more text. And more text. )"
    b" Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 628.8480 Td\r\n( And more text. And more text."
    b" And more text. And more text. And more ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 616.8960"
    b" Td\r\n( text. And more text. Boring, zzzzz. And more text. And more text. And )"
    b" Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 604.9440 Td\r\n( more text. And more text. And more"
    b" text. And more text. And more text. ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 592.9920 Td\r\n("
    b" And more text. And more text. ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 569.0880 Td\r\n( And"
    b" more text. And more text. And more text. And more text. And more ) Tj\r\nET\r\nBT\r\n/F1 0010"
    b" Tf\r\n69.2500 557.1360 Td\r\n( text. And more text. And more text. Even more. Continued on page"
    b" 2 ...) Tj\r\nET\r\n\nendstream\nendobj\nxref\n0 8\n0000000000 65535 f \n0000000009 00000 n"
    b" \n0000000068 00000 n \n0000000108 00000 n \n0000000251 00000 n \n0000000300 00000 n \n0000000407"
    b" 00000 n \n0000000437 00000 n \ntrailer\n<<\n/Size 8\n/Root 4 0 R\n/Info 2 0"
    b" R\n>>\nstartxref\n1563\n%%EOF\n"
)

PDF_BYTES_2 = (
    b"%PDF-1.3\n1 0 obj\n<<\n/Type /Pages\n/Count 1\n/Kids [ 3 0 R ]\n>>\nendobj\n2 0 obj\n<<\n/Producer"
    b" (PyPDF2)\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 1 0 R\n/Resources <<\n/Font <<\n/F1 5 0"
    b" R\n>>\n/ProcSet 6 0 R\n>>\n/MediaBox [ 0 0 1112 1792 ]\n/Contents 7 0 R\n>>\nendobj\n4 0"
    b" obj\n<<\n/Type /Catalog\n/Pages 1 0 R\n>>\nendobj\n5 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/Name"
    b" /F1\n/BaseFont /Helvetica\n/Encoding /WinAnsiEncoding\n>>\nendobj\n6 0 obj\n[ /PDF /Text ]\nendobj\n7"
    b" 0 obj\n<<\n/Length 676\n>>\nstream\n2 J\r\nBT\r\n0 0 0 rg\r\n/F1 0027 Tf\r\n57.3750 722.2800 Td\r\n("
    b" Simple PDF File 2 ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 688.6080 Td\r\n( ...continued from page"
    b" 1. Yet more text. And more text. And more text. ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 676.6560"
    b" Td\r\n( And more text. And more text. And more text. And more text. And more ) Tj\r\nET\r\nBT\r\n/F1"
    b" 0010 Tf\r\n69.2500 664.7040 Td\r\n( text. Oh, how boring typing this stuff. But not as boring as"
    b" watching ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 652.7520 Td\r\n( paint dry. And more text. And"
    b" more text. And more text. And more text. ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 640.8000"
    b" Td\r\n( Boring.  More, a little more text. The end, and just as well. ) Tj\r\nET\r\n\nendstream"
    b"\nendobj\nxref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000068 00000 n \n0000000108 00000"
    b" n \n0000000251 00000 n \n0000000300 00000 n \n0000000407 00000 n \n0000000437 00000 n \ntrailer"
    b"\n<<\n/Size 8\n/Root 4 0 R\n/Info 2 0 R\n>>\nstartxref\n1164\n%%EOF\n"
)
