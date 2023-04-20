#!/usr/bin/env python3
"""
Module for object detection using the EfficientDet model.

This module exports the following functions:
    - main(): draws bounding boxes around objects in an image using the EfficientDet lite object detection model
"""

"""Demo for object detection"""

__all__ = ["main"]

import numpy as np
import pkg_resources
import streamlit as st
from models.model import LABELS, get_size, inference
from PIL import Image, ImageDraw, ImageFont


def main():
    """Main function"""
    st.title("Demo")

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        size = get_size()
        original_width, original_height = img.size
        bboxes, class_ids, confs = inference(img)

        width_scale = original_width
        height_scale = original_height
        font_size = 30
        font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        for i, box in enumerate(bboxes[0]):
            box = [
                box[0] * height_scale,
                box[1] * width_scale,
                box[2] * height_scale,
                box[3] * width_scale,
            ]
            print(box)
            draw.rectangle(
                [(box[1], box[0]), (box[3], box[2])],
                outline=(0, 255, 0),
                width=3,
            )
            label = LABELS[int(np.squeeze(class_ids)[i])]
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((box[1], box[0] - text_height), label, font=font)
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
