import os
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_image  # Assuming preprocessing is in src

# Define class labels globally
CLASS_LABELS = {0: "Non-Cancerous", 1: "Cancerous"}

# Define the model directory
MODEL_DIR = "models"  # Adjust the path as needed


def load_model_by_name(model_name):
    """
    Load the specified model by name.

    Args:
        model_name (str): Name of the model file.

    Returns:
        model: Loaded Keras model.
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)


def make_prediction(image_path, model):
    """
    Preprocess the image and make a prediction.

    Args:
        image_path (str): Path to the image file.
        model: Loaded Keras model.

    Returns:
        tuple: Predicted class label and confidence percentage.
    """
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction) * 100)

        # Get the label
        label = CLASS_LABELS.get(predicted_class, "Unknown")
        return label, confidence
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
