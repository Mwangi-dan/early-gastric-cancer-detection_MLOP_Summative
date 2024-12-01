from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the saved model
model = load_model("models/refined_2.keras")

# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an uploaded image for model prediction.

    Args:
        file: The uploaded image file (e.g., from Flask request).
        target_size: Tuple, the size to resize the image to (width, height).

    Returns:
        A preprocessed numpy array ready for prediction.
    """
    img = load_img(image_path, target_size=target_size)  # Resize to target size
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array