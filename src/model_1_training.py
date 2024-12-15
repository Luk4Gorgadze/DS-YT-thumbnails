import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import create_image_data

# Load the data
data = pd.read_csv('storage/channel_videos_with_analysis.csv')

# Select relevant features and target variable
features = data[[
    'Title', 'Text', 'Language', 'Duration_minutes', 'Num_faces', 'Brightness',
    'Contrast', 'Saturation'
]]
target = data['Click_rate']

# Preprocessing
# Define the column transformer for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num', 'passthrough', [
                'Duration_minutes', 'Num_faces', 'Brightness', 'Contrast',
                'Saturation'
            ]
        ), ('cat', OneHotEncoder(handle_unknown='ignore'), ['Language'])
    ]
)

# Create a pipeline with preprocessing and model
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor), ('regressor', RandomForestRegressor())
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')


# Function to preprocess new image data
def preprocess_new_data(new_data):
    # Assuming new_data is a DataFrame similar to 'features'
    return new_data


image = Image.open('src/test_images/DachiPlaying.jpg')
new_image_data = create_image_data(image, 'Amazing gameplay', 'en', 2)

# Preprocess the new data
new_data_processed = preprocess_new_data(new_image_data)

# Make predictions for the new image
new_predictions = model.predict(new_data_processed)
print(f'Predicted Click Rate for new image: {new_predictions[0]}')
