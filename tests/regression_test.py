"""Module containing an object detection model using the EfficientDet architecture.
"""
import sys
import unittest

import numpy as np

sys.path.append("../efficientdet")
from efficientdet.demo import get_image
from efficientdet.models.model import inference


class TestEfficientDet(unittest.TestCase):
    """Test case for the EfficientDet object detection model."""

    def test_inference(self):
        """Test the `inference` function of the EfficientDet."""

        ci_true = np.array(
            [
                [
                    2,
                    17,
                    1,
                    7,
                    0,
                    1,
                    0,
                    14,
                    2,
                    2,
                    2,
                    0,
                    63,
                    0,
                    14,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    1,
                    63,
                    14,
                    63,
                ]
            ]
        )

        img = get_image()
        bboxes, class_ids, confs = inference(img)
        self.assertEqual(bboxes.shape, (1, 25, 4))
        self.assertEqual(class_ids.shape, (1, 25))
        self.assertEqual(confs.shape, (1, 25))
        self.assertTrue((class_ids == ci_true).all())


if __name__ == "__main__":
    unittest.main()
