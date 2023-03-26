"""Module containing an object detection model using the EfficientDet architecture.

"""
import unittest
import numpy as np
import sys

sys.path.append("../demo")
from demo.models.model import inference
from demo.demo import get_image


class TestEfficientDet(unittest.TestCase):
    """Test case for the EfficientDet object detection model."""

    def test_inference(self):
        """Test the `inference` function of the EfficientDet model.

        This test case checks that the `inference` function returns the correct output shapes and predicted class IDs for an example input image of a dog. 
        """

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
