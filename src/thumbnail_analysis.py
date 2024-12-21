import cv2
import easyocr
import numpy as np
import pandas as pd
import requests
from PIL import Image


def analyze_thumbnail(thumbnail_url):
    result = {
        'Num_faces': 0,
        'Text': '',
        'Brightness': 0,
        'Contrast': 0,
        'Saturation': 0,
    }

    try:
        img = Image.open(
            requests.get(
                thumbnail_url,
                stream=True,
                timeout=5,
            ).raw
        )
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Image loading error: {e}")
        return result

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            img_cv,
            scaleFactor=1.1,
            minNeighbors=5,
        )
        result['Num_faces'] = len(faces)
    except Exception as e:
        print(f"Face detection error: {e}")

    try:
        reader = easyocr.Reader(['en'], gpu=True)
        text_results = reader.readtext(img_cv)
        result['Text'] = " ".join([res[1] for res in text_results]).strip()
    except Exception as e:
        print(f"Text detection error: {e}")

    try:
        result['Brightness'] = np.mean(img_cv) / 255.0
        result['Contrast'] = np.std(img_cv) / 255.0
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        result['Saturation'] = np.mean(img_hsv[:, :, 1]) / 255.0
    except Exception as e:
        print(f"Image metrics error: {e}")

    return result


def perform_thumbnail_analysis(csv_file_path):
    channel_video_data = pd.read_csv(csv_file_path)

    thumbnail_analysis = channel_video_data.apply(
        lambda row: analyze_thumbnail(row['Thumbnail']),
        axis=1,
    )

    thumbnail_df = pd.DataFrame(thumbnail_analysis.tolist())

    channel_video_data = pd.concat(
        [
            channel_video_data,
            thumbnail_df,
        ],
        axis=1,
    )
    channel_video_data.to_csv(
        'storage/data/channel_videos_with_analysis.csv',
        index=False,
    )


def main():
    perform_thumbnail_analysis('storage/data/channel_videos.csv')


if __name__ == "__main__":
    main()
