import cv2
import numpy as np
import pandas as pd
import pytesseract
import requests
from PIL import Image, ImageEnhance


def analyze_thumbnail(thumbnail_url):
    """Analyze the thumbnail for faces, text, brightness, contrast, and saturation."""
    try:
        # Download the image with a timeout
        img = Image.open(
            requests.get(thumbnail_url, stream=True, timeout=5).raw
        )

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )

        # Preprocess the image for better OCR results
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(
            gray_img, 150, 255, cv2.THRESH_BINARY_INV
        )

        # Text detection
        text = pytesseract.image_to_string(binary_img, lang='eng')

        # Brightness, Contrast, and Saturation
        brightness = np.mean(img_cv) / 255.0  # Normalize to [0, 1]
        contrast = ImageEnhance.Contrast(img).enhance(1).getextrema()
        saturation = ImageEnhance.Color(img).enhance(1).getextrema()

        return {
            'num_faces': len(faces),
            'text': text.strip(),
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
        }
    except Exception as e:
        print(f"Error analyzing thumbnail: {e}")
        return {
            'num_faces': 0,
            'text': '',
            'brightness': 0,
            'contrast': (0, 0),
            'saturation': (0, 0),
        }


def perform_thumbnail_analysis(csv_file_path):
    """Perform thumbnail analysis by reading from a CSV file and save results to a new CSV file."""
    # Read the video data from the CSV file
    channel_video_data = pd.read_csv(csv_file_path)

    # Perform thumbnail analysis
    thumbnail_analysis = channel_video_data['Thumbnail'].apply(
        analyze_thumbnail
    )
    thumbnail_df = pd.DataFrame(thumbnail_analysis.tolist())

    # Combine video data with thumbnail analysis
    channel_video_data = pd.concat([channel_video_data, thumbnail_df], axis=1)

    # Save the combined data to a new CSV file
    output_file_path = 'storage/channel_videos_with_analysis.csv'
    channel_video_data.to_csv(output_file_path, index=False)
    print(f"Thumbnail analysis results saved to '{output_file_path}'")


def main():
    perform_thumbnail_analysis('storage/channel_videos.csv')


if __name__ == "__main__":
    main()
