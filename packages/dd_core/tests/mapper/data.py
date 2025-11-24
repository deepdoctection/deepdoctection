# -*- coding: utf-8 -*-
# File: xxx.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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

XFUND_RAW_LAYOUTLM_FEATURES = {
    "image_id": "c421a065-cfd4-3057-8d50-4b98e3c09810",
    "width": 1000,
    "height": 1000,
    "ann_ids": [
        "928d3b27-ba8e-30ed-9c1a-6230c501eea8",
        "149a57b7-4f50-377b-b793-422ff8d2a6a3",
        "b118fe90-d3e3-3f0c-a05b-f6765d01f1e7",
    ],
    "words": ["Akademisches", "Auslandsamt", "Bewerbungsformular"],
    "bbox": [[131.0, 52.0, 234.0, 66.0], [236.0, 53.0, 337.0, 67.0], [426.0, 117.0, 686.0, 138.0]],
    "dataset_type": "token_classification",
    "labels": [6, 6, 1],
}


XFUND_LAYOUTLM_FEATURES = {
        "image_ids": ["t74dfkh3-12gr-17d9-8e41-c4d134c0uzo4"],
        "width": [1000],
        "height": [1000],
        "ann_ids": [
            [
                "CLS",
                "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
                "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
                "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
                "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
                "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
                "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
                "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
                "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
                "SEP",
            ]
        ],
        "bbox": [
            [
                [0, 0, 0, 0],
                [325, 184, 578, 230],
                [325, 184, 578, 230],
                [325, 184, 578, 230],
                [325, 184, 578, 230],
                [586, 186, 834, 232],
                [586, 186, 834, 232],
                [586, 186, 834, 232],
                [586, 186, 834, 232],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [858, 413, 961, 482],
                [1000, 1000, 1000, 1000],
            ]
        ],
        "tokens": [
            [
                "CLS",
                "aka",
                "##de",
                "##mis",
                "##ches",
                "aus",
                "##lands",
                "##am",
                "##t",
                "be",
                "##wer",
                "##bu",
                "##ng",
                "##sf",
                "##or",
                "##mu",
                "##lar",
                "SEP",
            ]
        ],
        "input_ids": [
            [
                101,
                9875,
                3207,
                15630,
                8376,
                17151,
                8653,
                3286,
                2102,
                2022,
                13777,
                8569,
                3070,
                22747,
                2953,
                12274,
                8017,
                102,
            ]
        ],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        "token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    },


PRODIGY_DATAPOINT = {
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