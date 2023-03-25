#!/usr/bin/env python3
"""
Module for object detection using the EfficientDet model.

This module exports the following functions:
    - main(): draws bounding boxes around objects in an image using the EfficientDet lite object detection model
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from PIL import ImageDraw
from demo.model import inference, get_image, get_size, LABELS


def main():
    """
    Draws bounding boxes around objects in an image using the EfficientDet object detection model.

    This function loads an image from the file "dog.jpg", performs object detection, and draws bounding boxes around the detected objects in the image.
    Returns:
        None
    """
    img = get_image("dog.jpg")
    size = get_size()
    bboxes, class_ids, _ = inference(img)
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(bboxes[0]):
        draw.rectangle(
            [(box[1] * size, box[0] * size), (box[3] * size, box[2] * size)],
            outline=(0, 255, 0),
            width=2,
        )

        label = LABELS[int(np.squeeze(class_ids)[i]) + 1]
        text_bbox = draw.textbbox((0, 0), label)
        text_height = text_bbox[3] - text_bbox[1]
        draw.text((box[1] * size, box[0] * size - text_height), label)

    img.save("dog_result_tflite.jpg")

    #
    # Display the image on the screen
    # image = Image.open("dog_result_tflite.jpg")
    # image.show()


if __name__ == "__main__":
    main()
