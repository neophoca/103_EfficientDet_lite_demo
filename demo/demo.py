#!/usr/bin/env python3
import numpy as np
from PIL import ImageDraw
from demo.model import inference, get_image, get_size, LABELS


def main():
    img = get_image("dog.jpg")
    size = get_size()
    bboxes, class_ids, confs = inference(img)
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

    img.save("dog_result_tflite.jpg")

    #
    # Display the image on the screen
    # image = Image.open("dog_result_tflite.jpg")
    # image.show()


if __name__ == "__main__":
    main()
