#!/usr/bin/env python3
"""
This module provides an object detection model for detecting objects in images using TensorFlow Lite.

The module provides a function called 'inference' that takes an image as input and returns the bounding
boxes, class IDs, and confidence scores for the detected objects in the image.
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pkg_resources


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MODEL_CHOICE = 1
MODEL_TFLITE = "model_float32.tflite" + str(MODEL_CHOICE)

SIZE = [320, 384, 448, 512, 640][MODEL_CHOICE]


def get_size():
    """
    Returns the input size of the model.
    """
    return SIZE


LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


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


def inference(img):
    """
    Runs object detection on an input image using the TensorFlow Lite model.

    :param img: A PIL Image object representing the input image.
    :return: A tuple containing three arrays:
        - An array of bounding boxes, where each box is represented as [ymin, xmin, ymax, xmax]
        - An array of class IDs, where each ID is an integer representing the object class
        - An array of confidence scores, where each score is a float between 0 and 1 representing the confidence in the
          detection
    """
    with pkg_resources.resource_stream(__name__, MODEL_TFLITE) as model_file:
        interpreter = tf.lite.Interpreter(
            model_content=model_file.read(), num_threads=4
        )

    interpreter.allocate_tensors()

    img = img.resize((SIZE, SIZE))
    frame = np.array(img)
    frame = frame.reshape((1, SIZE, SIZE, 3))

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, frame.astype(np.float32))
    interpreter.invoke()

    bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])
    confs = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])

    return bboxes, class_ids, confs
