#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pkg_resources

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_choice = 3
model_tflite = "models/model_float32.tflite" + str(model_choice)

size = [320, 384, 448, 512, 640][model_choice]


def get_size():
    return size


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
    if pkg_resources.resource_exists(__name__, file_path):
        with pkg_resources.resource_stream(__name__, file_path) as image_file:
            image = Image.open(image_file)
            image.load()
    return image


def inference(img):
    with pkg_resources.resource_stream(__name__, model_tflite) as model_file:
        interpreter = tf.lite.Interpreter(
            model_content=model_file.read(), num_threads=4
        )

    interpreter.allocate_tensors()

    w, h = img.size
    img = img.resize((size, size))
    frame = np.array(img)
    frame = frame.reshape((1, size, size, 3))

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, frame.astype(np.float32))
    interpreter.invoke()

    bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])
    confs = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])

    return bboxes, class_ids, confs
