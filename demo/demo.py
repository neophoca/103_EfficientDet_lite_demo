#!/usr/bin/env python3
"""
Module for object detection using the EfficientDet model.

This module exports the following functions:
    - main(): draws bounding boxes around objects in an image using the EfficientDet lite object detection model
"""
import numpy as np
import pkg_resources
from PIL import ImageDraw, Image
from demo.models.model import inference, get_size, LABELS


def get_image(file_path):
    """
    Loads an image from a file path and returns it as a PIL Image object.

    :param file_path: The pat       h to the image file.
    :return: A PIL Image object.
    """
    image = None
    if pkg_resources.resource_exists(__name__, file_path):
        with pkg_resources.resource_stream(__name__, file_path) as image_file:
            image = Image.open(image_file)
            image.load()
    return image


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
