import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thumbnail_analysis import analyze_thumbnail


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the data with proper error handling
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Convert text columns
        text_columns = ['Title', 'Text', 'Language']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '')

        # Convert numeric columns
        numeric_columns = [
            'Duration_minutes', 'Num_faces', 'Brightness', 'Contrast',
            'Saturation'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'Click_rate' not in df.columns:
            raise ValueError("Click_rate column not found in the dataset")

        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def create_and_train_pipeline(data):
    """
    Create and train the prediction pipeline with categorical encoding
    """
    try:
        df = data.copy()

        # Drop unnecessary columns
        columns_to_drop = [
            'Channel_id',
            'ID',
            'Thumbnail',
            'Published_date',
            'Subscribers',
            'Likes',
            'Views',
        ]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns]
        )

        # Split features and target
        X = df.drop('Click_rate', axis=1)
        y = df['Click_rate']

        # Define features
        numeric_features = [
            'Duration_minutes',
            'Num_faces',
            'Brightness',
            'Contrast',
            'Saturation',
        ]
        categorical_features = ['Language']

        # Verify required columns
        missing_columns = [
            col for col in numeric_features + ['Title', 'Text'] +
            categorical_features if col not in X.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                (
                    'text_title',
                    TfidfVectorizer(
                        max_features=100,
                        stop_words='english',
                        strip_accents='unicode'
                    ),
                    'Title',
                ),
                (
                    'text_thumbnail',
                    TfidfVectorizer(
                        max_features=100,
                        stop_words='english',
                        strip_accents='unicode'
                    ),
                    'Text',
                ),
                (
                    'cat',
                    OneHotEncoder(
                        sparse_output=False, handle_unknown='ignore'
                    ),  # Changed this part
                    categorical_features
                ),
            ],
            sparse_threshold=0  # Force dense output
        )

        # Create pipeline
        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                (
                    'regressor',
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42,
                    )
                ),
            ]
        )

        # Fit pipeline
        pipeline.fit(X, y)
        return pipeline

    except Exception as e:
        print(f"Error training pipeline: {str(e)}")
        raise


def predict_click_rate(
    title: str,
    text: str,
    language: str,
    duration_minutes: int,
    num_faces: int,
    brightness: float,
    contrast: float,
    saturation: float,
    pipeline=None,
):
    """
    Predict click rate for a new thumbnail
    """
    try:
        if pipeline is None:
            model_path = 'storage/models/click_rate_predictor.joblib'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            pipeline = joblib.load(model_path)

        new_data = pd.DataFrame(
            {
                'Title': [str(title) if title is not None else ''],
                'Text': [str(text) if text is not None else ''],
                'Language': [str(language)],
                'Duration_minutes': [float(duration_minutes)],
                'Num_faces': [int(num_faces)],
                'Brightness': [float(brightness)],
                'Contrast': [float(contrast)],
                'Saturation': [float(saturation)],
            }
        )

        prediction = pipeline.predict(new_data)[0]
        return np.clip(prediction, 0, 1)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    # Load and train model if needed
    try:
        pipeline = joblib.load('storage/models/click_rate_predictor.joblib')
        print("Loaded existing model")
    except:
        print("Training new model...")
        data = load_and_preprocess_data(
            'storage/data/channel_videos_with_analysis.csv'
        )
        pipeline = create_and_train_pipeline(data)
        joblib.dump(pipeline, 'storage/models/click_rate_predictor.joblib')
        print("Model trained and saved")

    # Test prediction
    print("\nTesting prediction on a sample video...")
    try:
        test_image_path = 'src/test_images/DachiPlaying.jpg'

        thumbnail_data = analyze_thumbnail(test_image_path)
        predicted_rate = predict_click_rate(
            title='Amazing Gameplay',
            language='en',
            duration_minutes=200,
            num_faces=thumbnail_data['Num_faces'],
            brightness=thumbnail_data['Brightness'],
            contrast=thumbnail_data['Contrast'],
            saturation=thumbnail_data['Saturation'],
            text=thumbnail_data['Text'],
            pipeline=pipeline,
        )

        print(f'Predicted Click Rate: {predicted_rate:.4f}')
    except Exception as e:
        print(f"Error during testing: {str(e)}")
