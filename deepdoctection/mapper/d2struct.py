from typing import Dict, Union, Optional

from detectron2.structures import BoxMode
from ..utils.detection_types import JsonDict
from ..datapoint.image import Image
from ..mapper.maputils import curry


@curry
def image_to_d2_training(dp: Image, add_mask: bool = False) -> Optional[JsonDict]:
    """
    Maps an image to a standard dataset dict as described in
    https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html. Note, that the returned
    dict will not suffice for training as gt for RPN and anchors still need to be created.
    :param dp: Image
    :param add_mask: True is not implemented (yet).
    :return:
    """

    output: JsonDict = {}

    output["file_name"] = dp.location
    if dp.image is not None:
        output["image"] = dp.image.astype("float32")
    output["width"] = dp.width
    output["height"] = dp.height
    output["image_id"] = dp.image_id

    anns = dp.get_annotation()

    if not anns:
        return None

    annotations = []

    for ann in anns:
        mapped_ann: Dict[str,Union[str,int]] = {}
        mapped_ann["bbox_mode"] = BoxMode.XYXY_ABS
        mapped_ann["bbox"] = ann.bounding_box.to_list(mode="xyxy")
        mapped_ann["category_id"]=int(ann.category_id) - 1
        annotations.append(mapped_ann)

        if add_mask:
            raise NotImplementedError

    output["annotations"] = annotations

    return output

