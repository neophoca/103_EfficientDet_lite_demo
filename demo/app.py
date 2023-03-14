import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from .model import inference, get_size, LABELS

st.set_page_config(page_title="Demo")

st.title("Demo")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

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
