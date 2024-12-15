import cv2
import easyocr
import numpy as np
import pandas as pd


def create_image_data(image, title, language, duration):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    brightness = np.mean(img_cv) / 255.0
    reader = easyocr.Reader(['en'], gpu=True)
    text_results = reader.readtext(img_cv)
    text = " ".join([res[1] for res in text_results]).strip()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        img_cv,
        scaleFactor=1.1,
        minNeighbors=5,
    )
    contrast = np.std(img_cv) / 255.0
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    saturation = np.mean(img_hsv[:, :, 1]) / 255.0
    return pd.DataFrame(
        {
            'Title': [title],
            'Text': [text],
            'Language': [language],
            'Duration_minutes': [duration],
            'Num_faces': [len(faces)],
            'Brightness': [brightness],
            'Contrast': [contrast],
            'Saturation': [saturation]
        }
    )
