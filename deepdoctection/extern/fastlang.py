# -*- coding: utf-8 -*-
# File: fastlang.py

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
Deepdoctection wrappers for fasttext language detection models
"""

from typing import List

from ..utils.file_utils import Requirement, fasttext_available, get_fasttext_requirement
from .base import DetectionResult, LanguageDetector, PredictorBase

if fasttext_available():
    from fasttext import load_model  # type: ignore

FASTTEXT_GEO691_2 = {
    "__label__en": "eng",
    "__label__ru": "rus",
    "__label__de": "deu",
    "__label__fr": "fre",
    "__label__it": "ita",
    "__label__ja": "jpn",
    "__label__es": "spa",
    "__label__ceb": "ceb",
    "__label__tr": "tur",
    "__label__pt": "por",
    "__label__uk": "ukr",
    "__label__eo": "epo",
    "__label__pl": "pol",
    "__label__sv": "swe",
    "__label__nl": "dut",
    "__label__he": "heb",
    "__label__zh": "chi",
    "__label__hu": "hun",
    "__label__ar": "ara",
    "__label__ca": "cat",
    "__label__fi": "fin",
    "__label__cs": "cze",
    "__label__fa": "per",
    "__label__sr": "srp",
    "__label__el": "gre",
    "__label__vi": "vie",
    "__label__bg": "bul",
    "__label__ko": "kor",
    "__label__no": "nor",
    "__label__mk": "mac",
    "__label__ro": "rum",
    "__label__id": "ind",
    "__label__th": "tha",
    "__label__hy": "arm",
    "__label__da": "dan",
    "__label__ta": "tam",
    "__label__hi": "hin",
    "__label__hr": "hrv",
    "__label__sh": "",
    "__label__be": "bel",
    "__label__ka": "geo",
    "__label__te": "tel",
    "__label__kk": "kaz",
    "__label__war": "war",
    "__label__lt": "lit",
    "__label__gl": "glg",
    "__label__sk": "slo",
    "__label__bn": "ben",
    "__label__eu": "baq",
    "__label__sl": "slv",
    "__label__kn": "",
    "__label__ml": "mal",
    "__label__mr": "mar",
    "__label__et": "est",
    "__label__az": "aze",
    "__label__ms": "",
    "__label__sq": "alb",
    "__label__la": "lat",
    "__label__bs": "bos",
    "__label__nn": "nno",
    "__label__ur": "urd",
    "__label__lv": "",
    "__label__my": "",
    "__label__tt": "",
    "__label__af": "",
    "__label__oc": "",
    "__label__nds": "",
    "__label__ky": "",
    "__label__ast": "",
    "__label__tl": "",
    "__label__is": "",
    "__label__ia": "",
    "__label__si": "",
    "__label__gu": "",
    "__label__km": "",
    "__label__br": "",
    "__label__ba": "",
    "__label__uz": "",
    "__label__bo": "",
    "__label__pa": "",
    "__label__vo": "",
    "__label__als": "",
    "__label__ne": "",
    "__label__cy": "",
    "__label__jbo": "",
    "__label__fy": "",
    "__label__mn": "",
    "__label__lb": "",
    "__label__ce": "",
    "__label__ug": "",
    "__label__tg": "",
    "__label__sco": "",
    "__label__sa": "",
    "__label__cv": "",
    "__label__jv": "",
    "__label__min": "",
    "__label__io": "",
    "__label__or": "",
    "__label__as": "",
    "__label__new": "",
    "__label__ga": "",
    "__label__mg": "",
    "__label__an": "",
    "__label__ckb": "",
    "__label__sw": "",
    "__label__bar": "",
    "__label__lmo": "",
    "__label__yi": ":" "",
    "__label__arz": "",
    "__label__mhr": "",
    "__label__azb": "",
    "__label__sah": "",
    "__label__pnb": "",
    "__label__su": "",
    "__label__bpy": "",
    "__label__pms": "",
    "__label__ilo": "",
    "__label__wuu": "",
    "__label__ku": "",
    "__label__ps": "",
    "__label__ie": "",
    "__label__xmf": "",
    "__label__yue": "",
    "__label__gom": "",
    "__label__li": "",
    "__label__mwl": "",
    "__label__kw": "",
    "__label__sd": "",
    "__label__hsb": "",
    "__label__scn": "",
    "__label__gd": "",
    "__label__pam": "",
    "__label__bh": "",
    "__label__mai": "",
    "__label__vec": "",
    "__label__mt": "",
    "__label__dv": "",
    "__label__wa": "",
    "__label__mzn": "",
    "__label__am": "",
    "__label__qu": "",
    "__label__eml": "",
    "__label__cbk": "",
    "__label__tk": "",
    "__label__rm": "",
    "__label__os": "",
    "__label__vls": "",
    "__label__yo": "",
    "__label__lo": "",
    "__label__lez": "",
    "__label__so": "",
    "__label__myv": "",
    "__label__diq": "",
    "__label__mrj": "",
    "__label__dsb": "",
    "__label__frr": "",
    "__label__ht": "",
    "__label__gn": "",
    "__label__bxr": "",
    "__label__kv": "",
    "__label__sc": "",
    "__label__nah": "",
    "__label__krc": "",
    "__label__bcl": "",
    "__label__nap": "",
    "__label__gv": "",
    "__label__av": "",
    "__label__rue": "",
    "__label__xal": "",
    "__label__pfl": "",
    "__label__dty": "",
    "__label__hif": "",
    "__label__co": "",
    "__label__lrc": "",
    "__label__vep": "",
    "__label__tyv": "",
}


class FasttextLangDetector(LanguageDetector):
    """
    Fasttext language detector wrapper. Two models provided in the fasttext library can be used to identify languages.
    The background to the models can be found in the works:

    [1] Joulin A, Grave E, Bojanowski P, Mikolov T, Bag of Tricks for Efficient Text Classification

    [2] Joulin A, Grave E, Bojanowski P, Douze M, Jégou H, Mikolov T, FastText.zip: Compressing text classification
        models

    The models are distributed under the Creative Commons Attribution-Share-Alike License 3.0.
    (https://creativecommons.org/licenses/by-sa/3.0/)

    When loading the models via the ModelCatalog, the original and unmodified models are used.

    .. code-block:: python

        path_weights = ModelCatalog.get_full_path_weights("fasttext/lid.176.bin")
        lang_detector = FasttextLangDetector(path_weights)
        detection_result = lang_detector.predict("some text in some language")

    """

    def __init__(self, path_weights: str):
        """
        :param path_weights: path to model weights
        """

        super().__init__()
        self.path_weights = path_weights
        self.model = load_model(self.path_weights)

    def predict(self, text_string: str) -> DetectionResult:
        output = self.model.predict(text_string)
        return DetectionResult(text=FASTTEXT_GEO691_2[output[0][0]], score=output[1][0])

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_fasttext_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_weights)
