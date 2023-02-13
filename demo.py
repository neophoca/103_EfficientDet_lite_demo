import os
import tensorflow as tf
import pprint
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw
#import tflite_runtime.interpreter as tflite

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model_choice = 3
model_tflite = 'models/model_float32.tflite' + str(model_choice)

size = [320, 384, 448, 512, 640][model_choice]


LABELS = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

def structure_print():
    print('')
    print(f'model: {os.path.basename(model_tflite)}')
    print('')
    print('==INPUT============================================')
    pprint.pprint(interpreter.get_input_details())
    print('')
    print('==OUTPUT===========================================')
    pprint.pprint(interpreter.get_output_details())

interpreter = tf.lite.Interpreter(model_tflite, num_threads=4)
interpreter.allocate_tensors()
structure_print()


img = Image.open("dog.jpg")
w, h = img.size
img = img.resize((size, size))
frame = np.array(img)
frame = frame.reshape((1, size, size, 3))

input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, frame.astype(np.float32))
interpreter.invoke()

bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])
confs = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])

print(bboxes.shape)
print(bboxes)
print(class_ids.shape)
print(class_ids) # We need to add +1 to the index of the result.
print(confs.shape)
print(confs)


draw = ImageDraw.Draw(img)
for i, box in enumerate(bboxes[0]):
    draw.rectangle(
        [
            (box[1] * size, box[0] * size), 
            (box[3] * size, box[2] * size)
        ],
        outline=(0, 255, 0),
        width=2
    )
    label = LABELS[int(np.squeeze(class_ids)[i])+1]
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text((box[1] * size, box[0] * size - text_height), label)

img.save('dog_result_tflite.jpg')