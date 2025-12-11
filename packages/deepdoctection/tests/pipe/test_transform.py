

from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_array_equal

from dd_core.datapoint.image import Image
from dd_core.utils.identifier import get_uuid_from_str
from dd_core.utils.object_types import PageType

from deepdoctection.extern.base import DetectionResult, ImageTransformer
from deepdoctection.pipe.transform import SimpleTransformService



class TestSimpleTransformService:
    """
    Test SimpleTransformService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._transform_predictor = MagicMock(spec=ImageTransformer)
        self._transform_predictor.get_category_names = MagicMock(return_value=(PageType.ANGLE,))
        self._transform_predictor.name = "mock_transform"
        self._transform_predictor.model_id = get_uuid_from_str(self._transform_predictor.name)[:8]
        detect_result = DetectionResult()
        detect_result.new_h = 794  # type: ignore
        detect_result.new_w = 596  # type: ignore
        self._transform_predictor.predict = MagicMock(return_value=detect_result)
        self.simple_transform = SimpleTransformService(self._transform_predictor)


    def test_pass_datapoint(self, dp_image: Image) -> None:
        """
        test pass_datapoint
        """

        # Arrange
        np_output_img = np.ones((794, 596, 3), dtype=np.uint8) * 255
        self._transform_predictor.transform_image = MagicMock(return_value=np_output_img)

        # Act
        dp = self.simple_transform.pass_datapoint(dp_image)

        # Assert
        assert_array_equal(dp.image, np_output_img)  # type: ignore
        assert dp.width == 596
        assert dp.height == 794