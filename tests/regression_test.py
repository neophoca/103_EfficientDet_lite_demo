import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from demo import model


class TestEfficientDet(unittest.TestCase):
    def test_inference(self):
        img = demo.model.get_image("dog.jpg")
        bboxes, class_ids, confs = demo.model.inference(img)
        self.assertEqual(bboxes.shape, (100, 4))
        self.assertEqual(class_ids.shape, (100,))
        self.assertEqual(confs.shape, (100,))
        self.assertTrue(np.all(bboxes >= 0))
        self.assertTrue(np.all(bboxes <= 1))
        self.assertTrue(np.all(confs >= 0))
        self.assertTrue(np.all(confs <= 1))
