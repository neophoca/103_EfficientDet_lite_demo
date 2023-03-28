#!/usr/bin/env python3
"""
Module for object detection using the EfficientDet model.

This module exports the following functions:
    - main(): draws bounding boxes around objects in an image using the EfficientDet lite object detection model
"""

"""Demo for object detection"""

__all__ = ["main"]

import numpy as np
import streamlit as st
import pkg_resources
from PIL import Image, ImageDraw

from demo.models.model import LABELS, get_size, inference


def get_size():
    """Get size"""
    pass


def inference(image):
    """Perform inference"""
    pass


def main():
    """Main function"""
    st.set_page_config(page_title="Demo")
    st.title("Demo")

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        size = get_size()
        bboxes, class_ids, confs = inference(np.array(img))
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(bboxes[0]):
            draw.rectangle(
                [(box[1] * size, box[0] * size), (box[3] * size, box[2] * size)],
                outline=(0, 255, 0),
                width=2,
            )
            label = LABELS[int(np.squeeze(class_ids)[i]) + 1]
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((box[1] * size, box[0] * size - text_height), label)
        st.image(np.array(img), caption="Result", use_column_width=True)


def get_image(file_path="dog.jpg"):
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


if __name__ == "__main__":
    main()
