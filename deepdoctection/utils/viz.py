# -*- coding: utf-8 -*-
# File: viz.py

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
Visualisation utils. Copied and pasted from
"""

import base64
import os
import sys
from io import BytesIO
from typing import Any, Optional, Sequence, no_type_check

import numpy as np
import numpy.typing as npt
from lazy_imports import try_import
from numpy import float32, uint8

from .env_info import ENV_VARS_TRUE, auto_select_viz_library
from .error import DependencyError
from .file_utils import get_opencv_requirement, get_pillow_requirement
from .types import BGR, B64Str, PathLikeOrStr, PixelValues

with try_import() as cv2_import_guard:
    import cv2

with try_import() as pil_import_guard:
    from PIL import Image, ImageDraw


__all__ = ["draw_boxes", "interactive_imshow", "viz_handler"]

_COLORS = (
    np.array(
        [
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def random_color(rgb: bool = True, maximum: int = 255) -> tuple[int, int, int]:
    """
    Args:
        rgb: Whether to return RGB colors or BGR colors.
        maximum: Either 255 or 1.

    Returns:
        A tuple of three integers representing the color.
    """

    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return tuple(int(x) for x in ret)  # type: ignore


def draw_boxes(
    np_image: PixelValues,
    boxes: npt.NDArray[float32],
    category_names_list: Optional[list[Optional[str]]] = None,
    color: Optional[BGR] = None,
    font_scale: float = 1.0,
    rectangle_thickness: int = 4,
    box_color_by_category: bool = True,
    show_palette: bool = True,
) -> PixelValues:
    """
    Draw bounding boxes with category names into image.

    Args:
        np_image: Image as `np.ndarray`.
        boxes: A numpy array of shape Nx4 where each row is `[x1, y1, x2, y2]`.
        category_names_list: List of N category names.
        color: A 3-tuple BGR color (in range `[0, 255]`).
        font_scale: Font scale of text box.
        rectangle_thickness: Thickness of bounding box.
        box_color_by_category: Whether to color boxes by category.
        show_palette: Whether to show a color palette of the categories.

    Returns:
        A new image as `np.ndarray`.

    Raises:
        AssertionError: If the length of `category_names_list` does not match the number of boxes, or if any area is
                        not positive, or if boxes are out of image bounds.
    """
    if color is not None:
        box_color_by_category = False

    category_to_color = {}
    if box_color_by_category and category_names_list is not None:
        category_names = set(category_names_list)
        category_to_color = {category: random_color() for category in category_names}

    boxes = np.array(boxes, dtype="int32")
    if category_names_list is not None:
        assert len(category_names_list) == len(boxes), f"{len(category_names_list)} != {len(boxes)}"
    else:
        category_names_list = [None] * len(boxes)
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)  # draw large ones first
    assert areas.min() > 0, areas.min()

    # allow equal, because we are not very strict about rounding error here
    assert (
        boxes[:, 0].min() >= 0
        and boxes[:, 1].min() >= 0
        and boxes[:, 2].max() <= np_image.shape[1]
        and boxes[:, 3].max() <= np_image.shape[0]
    ), f"Image shape: {str(np_image.shape)}\n Boxes:\n{str(boxes)}"

    np_image = np_image.copy()

    if np_image.ndim == 2 or (np_image.ndim == 3 and np_image.shape[2] == 1):
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    for i in sorted_inds:
        box = boxes[i, :]
        choose_color = category_to_color.get(category_names_list[i]) if category_to_color is not None else color
        if choose_color is None:
            choose_color = random_color()
        if category_names_list[i] is not None:
            np_image = viz_handler.draw_text(
                np_image, (box[0], box[1]), category_names_list[i], color=choose_color, font_scale=font_scale
            )
        np_image = viz_handler.draw_rectangle(
            np_image, (box[0], box[1], box[2], box[3]), choose_color, rectangle_thickness
        )

    # draw a (very ugly) color palette
    if show_palette:
        y_0 = np_image.shape[0]
        for category, col in category_to_color.items():
            if category is not None:
                np_image = viz_handler.draw_text(
                    np_image,
                    (np_image.shape[1], y_0),
                    category,
                    color=col,
                    font_scale=font_scale,
                    rectangle_thickness=rectangle_thickness,
                )
                _, text_h = viz_handler.get_text_size(category, font_scale * 2)
                y_0 = y_0 - int(1 * text_h)

    return np_image


@no_type_check
def interactive_imshow(img: PixelValues) -> None:
    """
    Display an image in a pop-up window.

    Args:
        img: An image (expect BGR) to show.

    Example:
        ```python
        interactive_imshow(img)
        ```
    """
    viz_handler.interactive_imshow(img)


class VizPackageHandler:
    """
    A handler for the image processing libraries PIL or OpenCV. Explicit use of the libraries is not intended.
    If the environ.ment variable USE_DD_OPENCV=True is set, only the CV2 functions will be used via the handler.
    The default library is PIL. Compared to OpenCV, PIL is somewhat slower (this applies to reading and writing
    image files), which can lead to a bottleneck during training, especially if the loading is not parallelized
    """

    PACKAGE_FUNCS = {
        "cv2": {
            "read_image": "_cv2_read_image",
            "write_image": "_cv2_write_image",
            "convert_np_to_b64": "_cv2_convert_np_to_b64",
            "convert_b64_to_np": "_cv2_convert_b64_to_np",
            "resize": "_cv2_resize",
            "get_text_size": "_cv2_get_text_size",
            "draw_rectangle": "_cv2_draw_rectangle",
            "draw_text": "_cv2_draw_text",
            "interactive_imshow": "_cv2_interactive_imshow",
            "encode": "_cv2_encode",
            "rotate_image": "_cv2_rotate_image",
            "convert_bytes_to_np": "_cv2_convert_bytes_to_np",
        },
        "pillow": {
            "read_image": "_pillow_read_image",
            "write_image": "_pillow_write_image",
            "convert_np_to_b64": "_pillow_convert_np_to_b64",
            "convert_b64_to_np": "_pillow_convert_b64_to_np",
            "resize": "_pillow_resize",
            "get_text_size": "_pillow_get_text_size",
            "draw_rectangle": "_pillow_draw_rectangle",
            "draw_text": "_pillow_draw_text",
            "interactive_imshow": "_pillow_interactive_imshow",
            "encode": "_pillow_encode",
            "rotate_image": "_pillow_rotate_image",
            "convert_bytes_to_np": "_pillow_convert_bytes_to_np",
        },
    }

    def __init__(self) -> None:
        """Selecting the image processing library and fonts"""
        package = self._select_package()
        self.pkg_func_dict: dict[str, str] = {}
        self.font = None
        self._set_vars(package)

    @staticmethod
    def _select_package() -> str:
        """
        `USE_DD_OPENCV` has priority and will enforce to use OpenCV.
        Otherwise it will use Pillow as default package.

        Returns:
            Either 'pillow' or 'cv2'.

        Raises:
            EnvironmentError: If both `USE_DD_OPENCV` and `USE_DD_PILLOW` are set to `False` or `True`.
            DependencyError: If the required package is not available.
        """
        maybe_cv2 = "cv2" if os.environ.get("USE_DD_OPENCV", "False") in ENV_VARS_TRUE else None
        maybe_pil = "pillow" if os.environ.get("USE_DD_PILLOW", "True") in ENV_VARS_TRUE else None

        if not maybe_cv2 and not maybe_pil:
            raise EnvironmentError(
                "Both variables USE_DD_OPENCV and USE_DD_PILLOW are set to False. Please set only one of them to True"
            )
        if maybe_cv2 and maybe_pil:
            raise EnvironmentError(
                "Both variables USE_DD_OPENCV and USE_DD_PILLOW are set to True. Please set one of them to False."
            )

        # USE_DD_OPENCV has priority
        if maybe_cv2:
            requirements = get_opencv_requirement()
            if not requirements[1]:
                raise DependencyError(requirements[2])
            return maybe_cv2

        requirements = get_pillow_requirement()
        if not requirements[1]:
            raise DependencyError(requirements[2])
        return "pillow"

    def _set_vars(self, package: str) -> None:
        self.pkg_func_dict = self.PACKAGE_FUNCS[package]
        if package == "pillow":
            image = Image.fromarray(np.uint8(np.ones((1, 1, 3))))
            self.font = ImageDraw.ImageDraw(image).getfont()
        else:
            self.font = cv2.FONT_HERSHEY_SIMPLEX  # type: ignore

    def refresh(self) -> None:
        """
        Refresh the `viz_handler` setting. Useful if you change the environment variable at runtime and want to take
        account of the changes.

        Example:
            ```python
            os.env["USE_DD_OPENCV"]="True"
            viz_handler.refresh()
            ```

        Returns:
            None
        """
        package = self._select_package()
        self._set_vars(package)

    def read_image(self, path: PathLikeOrStr) -> PixelValues:
        """
        Reading an image from file and returning a `np.array`.

        Args:
            path: Use `/path/to/dir/file_name.[suffix]`.

        Returns:
            Image as `np.array`.
        """
        return getattr(self, self.pkg_func_dict["read_image"])(path)

    @staticmethod
    def _cv2_read_image(path: PathLikeOrStr) -> PixelValues:
        return cv2.imread(os.fspath(path), cv2.IMREAD_COLOR).astype(np.uint8) # type: ignore

    @staticmethod
    def _pillow_read_image(path: PathLikeOrStr) -> PixelValues:
        with Image.open(os.fspath(path)).convert("RGB") as image:
            np_image = np.array(image)[:, :, ::-1]
        return np_image

    def write_image(self, path: PathLikeOrStr, image: PixelValues) -> None:
        """
        Writing an image as `np.array` to a file.

        Args:
            path: Use `/path/to/dir/file_name.[suffix]`.
            image: Pixel values as `np.array`.

        Returns:
            None
        """
        return getattr(self, self.pkg_func_dict["write_image"])(path, image)

    @staticmethod
    def _cv2_write_image(path: PathLikeOrStr, image: PixelValues) -> None:
        cv2.imwrite(os.fspath(path), image)

    @staticmethod
    def _pillow_write_image(path: PathLikeOrStr, image: PixelValues) -> None:
        pil_image = Image.fromarray(np.uint8(image[:, :, ::-1]))
        pil_image.save(os.fspath(path))

    def encode(self, np_image: PixelValues) -> bytes:
        """
        Converting an image as `np.array` into a b64 representation.

        Args:
            np_image: Image as `np.array`.

        Returns:
            Image as bytes.
        """
        return getattr(self, self.pkg_func_dict["encode"])(np_image)

    @staticmethod
    def _cv2_encode(np_image: PixelValues) -> bytes:
        np_encode = cv2.imencode(".png", np_image)
        b_image = np_encode[1].tobytes()
        return b_image

    @staticmethod
    def _pillow_encode(np_image: PixelValues) -> bytes:
        buffered = BytesIO()
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue()

    def convert_np_to_b64(self, image: PixelValues) -> str:
        """
        Converting an image given as `np.array` into a b64 encoded string.

        Args:
            image: Image as `np.array`.

        Returns:
            b64 encoded string.
        """
        return getattr(self, self.pkg_func_dict["convert_np_to_b64"])(image)

    @staticmethod
    def _cv2_convert_np_to_b64(image: PixelValues) -> str:
        np_encode = cv2.imencode(".png", image)
        return base64.b64encode(np_encode[1]).decode("utf-8")  # type: ignore

    @staticmethod
    def _pillow_convert_np_to_b64(np_image: PixelValues) -> str:
        buffered = BytesIO()
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def convert_b64_to_np(self, image: B64Str) -> PixelValues:
        """
        Converting an image as b64 encoded string into `np.array`.

        Args:
            image: b64 encoded string.

        Returns:
            `np.array`.
        """
        return getattr(self, self.pkg_func_dict["convert_b64_to_np"])(image)

    @staticmethod
    def _cv2_convert_b64_to_np(image: B64Str) -> PixelValues:
        np_array = np.fromstring(base64.b64decode(image), np.uint8)  # type: ignore
        np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR).astype(np.float32) # type: ignore
        return np_array.astype(uint8)

    @staticmethod
    def _pillow_convert_b64_to_np(image: B64Str) -> PixelValues:
        array = base64.b64decode(image)
        im_file = BytesIO(array)
        pil_image = Image.open(im_file)
        return np.array(pil_image)[:, :, ::-1]

    def convert_bytes_to_np(self, image_bytes: bytes) -> PixelValues:
        """
        Converting an image as bytes into `np.array`.

        Args:
            image_bytes: Image as bytes.

        Returns:
            Image as `np.array`.
        """
        return getattr(self, self.pkg_func_dict["convert_bytes_to_np"])(image_bytes)

    @staticmethod
    def _cv2_convert_bytes_to_np(image_bytes: bytes) -> PixelValues:
        np_array = np.frombuffer(image_bytes, np.uint8)
        np_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return np_image  # type: ignore

    @staticmethod
    def _pillow_convert_bytes_to_np(image_bytes: bytes) -> PixelValues:
        image = Image.open(BytesIO(image_bytes))
        np_image = np.array(image)
        return np_image

    def resize(self, image: PixelValues, width: int, height: int, interpolation: str) -> PixelValues:
        """
        Resize a given image to new width, height. Specifying an interpolation method is required. Depending on the
         chosen image library use one of the following:

        PIL: NEAREST, BOX, BILINEAR, BICUBIC, VIZ (available for CV2 as well)
        CV2: INTER_NEAREST, INTER_LINEAR, INTER_AREA, VIZ

        Args:
            image: Image as `np.array`.
            width: The new image width.
            height: The new image height.
            interpolation: Interpolation method as string.

        Returns:
            Resized image as `np.array`.
        """
        return getattr(self, self.pkg_func_dict["resize"])(image, width, height, interpolation)

    @staticmethod
    def _cv2_resize(image: PixelValues, width: int, height: int, interpolation: str) -> PixelValues:
        intpol_method_dict = {
            "INTER_NEAREST": cv2.INTER_NEAREST,
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "INTER_AREA": cv2.INTER_AREA,
            "VIZ": cv2.INTER_LINEAR,
        }
        return cv2.resize(image, dsize=(width, height), interpolation=intpol_method_dict[interpolation]).astype(
            np.uint8
        )

    @staticmethod
    def _pillow_resize(image: PixelValues, width: int, height: int, interpolation: str) -> PixelValues:
        intpol_method_dict = {
            "NEAREST": Image.Resampling.NEAREST,
            "BOX": Image.Resampling.BOX,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
            "VIZ": Image.Resampling.BILINEAR,
        }
        pil_image = Image.fromarray(np.uint8(image[:, :, ::-1]))
        pil_image_resized = pil_image.resize(
            size=(width, height), resample=intpol_method_dict[interpolation], box=None, reducing_gap=None
        )
        return np.array(pil_image_resized)[:, :, ::-1]

    def get_text_size(self, text: str, font_scale: float) -> tuple[int, int]:
        """
        Return the text size for a given font scale.

        Args:
            text: Text as string.
            font_scale: Scale.

        Returns:
            A tuple with width and height of the text.
        """
        return getattr(self, self.pkg_func_dict["get_text_size"])(text, font_scale)

    def _cv2_get_text_size(self, text: str, font_scale: float) -> tuple[int, int]:
        ((width, height), _) = cv2.getTextSize(text, self.font, font_scale, 1)  # type: ignore
        return width, height

    def _pillow_get_text_size(self, text: str, font_scale: float) -> tuple[int, int]:  # pylint: disable=W0613
        _, _, width, height = self.font.getbbox(text)  # type: ignore
        return width, height

    def draw_rectangle(
        self, np_image: PixelValues, box: tuple[Any, Any, Any, Any], color: tuple[int, int, int], thickness: int
    ) -> PixelValues:
        """
        Drawing a rectangle into an image with a given color (b,g,r) and given thickness.

        Args:
            np_image: Image.
            box: Box (x_min, y_min, x_max, y_max).
            color: (b,g,r) between 0 and 255.
            thickness: Pixel width of the rectangle lines.

        Returns:
            Image with rectangle.
        """
        return getattr(self, self.pkg_func_dict["draw_rectangle"])(np_image, box, color, thickness)

    @staticmethod
    def _cv2_draw_rectangle(
        np_image: PixelValues, box: tuple[Any, Any, Any, Any], color: Sequence[int], thickness: int
    ) -> PixelValues:
        cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=thickness)
        return np_image

    @staticmethod
    def _pillow_draw_rectangle(
        np_image: PixelValues, box: tuple[Any, Any, Any, Any], color: Sequence[int], thickness: int
    ) -> PixelValues:
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(box, outline=color, width=thickness)  # type: ignore
        np_image = np.array(pil_image)[:, :, ::-1]
        return np_image

    def draw_text(
        self,
        np_image: PixelValues,
        pos: tuple[Any, Any],
        text: str,
        color: tuple[int, int, int],
        font_scale: float,
        rectangle_thickness: int = 1,
    ) -> PixelValues:
        """
        Drawing a text into a numpy image. The result will differ between PIL and CV2 (and will not look that good
        when using PIL).

        Args:
            np_image: Image.
            pos: x_min, y_min position of the starting point of the text.
            text: Text string.
            color: `(b,g,r)` between 0 and 255.
            font_scale: Scale of font. This will only be used within an OpenCV framework.
            rectangle_thickness: Thickness of the rectangle border.

        Returns:
            Image with text.
        """
        return getattr(self, self.pkg_func_dict["draw_text"])(
            np_image, pos, text, color, font_scale, rectangle_thickness
        )

    def _cv2_draw_text(
        self,
        np_image: PixelValues,
        pos: tuple[Any, Any],
        text: str,
        color: tuple[int, int, int],
        font_scale: float,
        rectangle_thickness: int,
    ) -> PixelValues:
        np_image = np_image.astype(np.uint8)
        x_0, y_0 = int(pos[0]), int(pos[1])
        # Compute text size.
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w, text_h = viz_handler.get_text_size(text, font_scale)
        # Place text background.
        if x_0 + text_w > np_image.shape[1]:
            x_0 = np_image.shape[1] - text_w
        if y_0 - int(1.15 * text_h) < 0:
            y_0 = int(1.15 * text_h)
        back_top_left = x_0, y_0 - int(1.3 * text_h)
        back_bottom_right = x_0 + text_w, y_0
        np_image = self.draw_rectangle(
            np_image,
            (back_top_left[0], back_top_left[1], back_bottom_right[0], back_bottom_right[1]),
            color,
            rectangle_thickness,
        )
        # Show text.
        text_bottomleft = x_0, y_0 - int(0.25 * text_h)
        cv2.putText(np_image, text, text_bottomleft, font, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return np_image

    @staticmethod
    def _pillow_draw_text(
        np_image: PixelValues,
        pos: tuple[Any, Any],
        text: str,
        color: tuple[int, int, int],  # pylint: disable=W0613
        font_scale: float,  # pylint: disable=W0613
        rectangle_thickness: int,  # pylint: disable=W0613
    ) -> PixelValues:
        # using PIL default font size that does not scale to larger image sizes.
        # Compare with https://github.com/python-pillow/Pillow/issues/6622
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        draw = ImageDraw.Draw(pil_image)
        draw.text(pos, text, fill=(0, 0, 0), anchor="lb")
        return np.array(pil_image)[:, :, ::-1]

    def interactive_imshow(self, np_image: PixelValues) -> None:
        """
        Displaying an image in a separate window.

        Args:
            np_image: Image as `np.array`.

        Returns:
            None
        """
        return getattr(self, self.pkg_func_dict["interactive_imshow"])(np_image)

    def _cv2_interactive_imshow(self, np_image: PixelValues) -> None:
        name = "q, x: quit / s: save"
        cv2.imshow(name, np_image)

        key = cv2.waitKey(-1)
        while key >= 128:
            key = cv2.waitKey(-1)
        key = chr(key & 0xFF)  # type: ignore

        if key == "q":
            cv2.destroyWindow(name)
        elif key == "x":
            sys.exit()
        elif key == "s":
            cv2.imwrite("out.png", np_image)
        elif key in ["+", "="]:
            np_image = cv2.resize(np_image, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
            self._cv2_interactive_imshow(np_image)
        elif key == "-":
            np_image = cv2.resize(np_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
            self._cv2_interactive_imshow(np_image)

    @staticmethod
    def _pillow_interactive_imshow(np_image: PixelValues) -> None:
        name = "q, x: quit / s: save"
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        pil_image.show(name)

    def rotate_image(self, np_image: PixelValues, angle: float) -> PixelValues:
        """
        Rotating an image by some angle.

        Args:
            np_image: Image as `np.array`.
            angle: Angle to rotate.

        Returns:
            Rotated image as `np.array`.
        """
        return getattr(self, self.pkg_func_dict["rotate_image"])(np_image, angle)

    @staticmethod
    def _cv2_rotate_image(np_image: PixelValues, angle: float) -> PixelValues:
        # copy & paste from https://stackoverflow.com/questions/43892506
        # /opencv-python-rotate-image-without-cropping-sides

        height, width = np_image.shape[:2]
        image_center = (width / 2, height / 2)
        rotation_mat = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        np_image = cv2.warpAffine(
            src=np_image,
            M=rotation_mat,
            dsize=(bound_w, bound_h),
        ).astype(np.uint8)

        return np_image

    @staticmethod
    def _pillow_rotate_image(np_image: PixelValues, angle: float) -> PixelValues:
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        pil_image_rotated = pil_image.rotate(angle, expand=True)
        return np.array(pil_image_rotated)[:, :, ::-1]


auto_select_viz_library()
viz_handler = VizPackageHandler()
