import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_style = ImageFont.truetype("simsun.ttc", text_size, encoding="utf-8")
    draw.text(position, text, text_color, font=font_style)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)