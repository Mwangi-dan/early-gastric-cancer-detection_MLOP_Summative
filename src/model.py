import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np


class GastricCancerPredictor:
    def __init__(self, data_dir, model_dir, target_size=(128, 128), batch_size=32):
        """
        Initializes the GastricCancerPredictor class.
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.train_dir = None
        self.val_dir = None
        self.model = None
        self.history = None

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_dataset(self, test_size=0.2):
        """
        Dynamically split the dataset into training and validation directories.

        Args:
            test_size (float): Proportion of data to allocate for validation.
        """
        classes = ['cancerous', 'non-cancerous']
        output_dir = os.path.join(self.data_dir, "split_data")

        # Create train/validation directories
        for split in ['train', 'validation']:
            for class_name in classes:
                os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

        # Split each class
        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            images = os.listdir(class_dir)
            train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)

            # Copy files to respective directories
            for img in train_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'train', class_name, img))
            for img in val_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'validation', class_name, img))

        # Update train and validation directories
        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "validation")

    def load_data(self):
        """
        Load the training and validation datasets using Keras ImageDataGenerator.
        """
        if not self.train_dir or not self.val_dir:
            raise ValueError("Training and validation directories are not set. Run prepare_dataset() first.")

        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        self.validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

    def build_model(self):
        """
        Build a Convolutional Neural Network (CNN) model.
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.target_size, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=10):
        """
        Train the model on the training dataset.
        """
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs
        )

    def evaluate_model(self):
        """
        Evaluate the model on the validation dataset.
        """
        self.y_test = self.validation_generator.classes  # True labels from the validation generator
        self.y_pred = (self.model.predict(self.validation_generator) > 0.5).astype(int).flatten()  # Predicted classes
        
        val_loss, val_accuracy = self.model.evaluate(self.validation_generator)
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}
    

    def plot_training_history(self, output_path="static/plots/training_history.png"):
        """
        Plot the training and validation accuracy and loss, and save the plot to a file.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the plot to the specified path
        plt.savefig(output_path)
        plt.close()  # Close the plot to free up memory


    def generate_confusion_matrix(self, output_path="static/plots/confusion_matrix.png"):
        """
        Generate a confusion matrix for the test data and save it as an image file.
        """
        if self.y_test is None or self.y_pred is None:
            raise ValueError("Test labels and predictions are required to generate the confusion matrix.")

        # Ensure y_test is converted to a flat list (if it's an array)
        if not isinstance(self.y_test, (list, np.ndarray)):
            self.y_test = np.array(self.y_test)

        # Compute the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Extract unique labels from both y_test and y_pred to ensure proper ordering
        unique_labels = sorted(set(np.concatenate((self.y_test, self.y_pred))))

        # Create the plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the plot to the specified path
        plt.savefig(output_path)
        plt.close()  # Close the plot to free up memory


    def save_model(self):
        """
        Save the trained model to the model directory with a unique name.

        The model is saved with a timestamp in the filename to avoid overwriting.
        """
        if not self.model:
            raise ValueError("No model is available to save. Train the model first.")

        # Generate a unique filename using a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        model_filename = f"model_{timestamp}.keras"

        # Full path for the model
        model_path = os.path.join(self.model_dir, model_filename)

        # Save the model
        self.model.save(model_path)
        print(f"Model saved as {model_filename}")
        return model_filename  # Return the filename for reference

if __name__ == "__main__":
    # Example usage
    data_dir = "uploaded_datasets"  # Replace with your dataset path
    model_dir = "models"

    predictor = GastricCancerPredictor(data_dir, model_dir)

    # Prepare the dataset
    predictor.prepare_dataset()

    # Load the data
    predictor.load_data()

    # Build, train, and evaluate the model
    predictor.build_model()
    predictor.train_model(epochs=10)
    predictor.evaluate_model()

    # Plot training history
    predictor.plot_training_history()
