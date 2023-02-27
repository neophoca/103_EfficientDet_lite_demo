import unittest
from demo import model
import numpy as np


class TestEfficientDet(unittest.TestCase):
    def test_inference(self):
        ci_true = np.array(
            [
                [
                    1,
                    17,
                    7,
                    63,
                    63,
                    63,
                    63,
                    2,
                    71,
                    13,
                    1,
                    63,
                    1,
                    63,
                    2,
                    63,
                    63,
                    63,
                    63,
                    63,
                    63,
                    1,
                    2,
                    63,
                    2,
                ]
            ]
        )
        img = model.get_image("dog.jpg")
        bboxes, class_ids, confs = model.inference(img)
        self.assertEqual(bboxes.shape, (1, 25, 4))
        self.assertEqual(class_ids.shape, (1, 25))
        self.assertEqual(confs.shape, (1, 25))
        self.assertTrue((class_ids == ci_true).all())


if __name__ == "__main__":
    unittest.main()
