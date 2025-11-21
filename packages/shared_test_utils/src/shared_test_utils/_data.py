# -*- coding: utf-8 -*-
# File: _data.py

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


PDF_BYTES =  (
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


PRODIGY_SAMPLE = {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAiCAIAAAA24aWuAAAAWElEQVQ4EZXBAQEAAAABIP6P"
    "zgZV5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5DTzFG"
    "W9r8aRmwAAAABJRU5ErkJggg==",
    "text": "99999984_310518_J_1_150819_page_252.png",
    "spans": [
        {
            "label": "table",
            "x": 100,
            "y": 223.7,
            "w": 1442,
            "h": 1875.3,
            "type": "rect",
            "points": [[1, 2.7], [1, 29], [15, 29], [15, 2.7]],
        },
        {
            "label": "title",
            "x": 1181.6,
            "y": 99.2,
            "w": 364.6,
            "h": 68.8,
            "type": "rect",
            "points": [[11.6, 9.2], [11.6, 18], [16.2, 18], [16.2, 9.2]],
        },
    ],
    "_input_hash": -2030256107,
    "_task_hash": 1310891334,
    "_view_id": "image_manual",
    "width": 1650,
    "height": 2350,
    "answer": "reject",
}

XFUND_SAMPLE = {
    "lang": "de",
    "version": "0.1",
    "split": "val",
    "documents": [
        {
            "img": {"fname": "/path/to/dir/de_val_0.jpg", "width": 2480, "height": 3508},
            "id": "de_val_0",
            "uid": "09006d3b9d97e3797ac9b59464f6a8f487da6ad4176d0fbd53caa9ecf1a7bca4",
            "document": [
                {
                    "box": [325, 183, 833, 231],
                    "text": "Akademisches Auslandsamt",
                    "label": "other",
                    "words": [
                        {"box": [325, 184, 578, 230], "text": "Akademisches"},
                        {"box": [586, 186, 834, 232], "text": "Auslandsamt"},
                    ],
                    "linking": [],
                    "id": 0,
                },
                {
                    "box": [1057, 413, 1700, 483],
                    "text": "Bewerbungsformular",
                    "label": "header",
                    "words": [{"box": [1058, 413, 1701, 482], "text": "Bewerbungsformular"}],
                    "linking": [],
                    "id": 15,
                },
            ],
        }
    ],
}

