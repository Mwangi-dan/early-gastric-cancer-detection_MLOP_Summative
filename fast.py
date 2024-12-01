from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import shutil
import numpy as np
from src.preprocessing import preprocess_image, model
from src.predict import load_model_by_name, make_prediction
import os
import zipfile
import shutil
from src.model import GastricCancerPredictor
from tensorflow.keras.models import load_model
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the "static/plots" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# modle directories
UPLOAD_DIR = "uploaded_datasets"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Define the class labels
CLASS_LABELS = {0: "Non-Cancerous", 1: "Cancerous"}


@app.post("/predict/")
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form(...),
):
    """
    Perform prediction using the specified model.
    """
    try:
        # Validate model name
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name must be provided.")

        # Load the selected model
        model = load_model_by_name(model_name)

        # Save the uploaded image temporarily
        temp_file = f"temp_{image.filename}"
        with open(temp_file, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Perform the prediction
        label, confidence = make_prediction(temp_file, model)

        # Cleanup temporary file
        os.remove(temp_file)

        # Return the result
        return {
            "label": label,
            "confidence": confidence,
        }
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/train-model/")
async def train_model(zip_file: UploadFile = File(...)):
    try:
        # Save the uploaded zip file
        zip_path = os.path.join(UPLOAD_DIR, zip_file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)

        # Extract the zip file
        extract_path = os.path.join(UPLOAD_DIR, os.path.splitext(zip_file.filename)[0])
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Validate folder structure
        subdirs = os.listdir(extract_path)
        if "cancerous" not in subdirs or "non-cancerous" not in subdirs:
            shutil.rmtree(extract_path)
            raise HTTPException(
                status_code=400,
                detail="The uploaded folder must contain 'cancerous' and 'non-cancerous' subdirectories.",
            )

        # Train the model
        predictor = GastricCancerPredictor(data_dir=extract_path, model_dir=MODEL_DIR)

        # Run dataset preparation before loading data
        predictor.prepare_dataset()

        # Load data and proceed
        predictor.load_data()
        predictor.build_model()
        predictor.train_model(epochs=10)

        # Evaluate the model
        evaluation = predictor.evaluate_model()
        saved_model = predictor.save_model()
        predictor.plot_training_history()
        predictor.generate_confusion_matrix()

        return {
            "message": "Model trained successfully!",
            "validation_accuracy": evaluation["val_accuracy"],
            "validation_loss": evaluation["val_loss"],
            "saved_model": saved_model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/list-models/")
async def list_models():
    """
    List all available models in the model directory.
    """
    try:
        model_files = [
            f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")
        ]
        if not model_files:
            return {"message": "No models available."}

        return {"models": model_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/get-training-plot/")
async def get_training_plot():
    plot_path = "static/plots/training_history.png"
    if not os.path.exists(plot_path):
        return {"error": "Training plot not found. Train the model first."}
    return FileResponse(plot_path)


@app.get("/get-confusion-matrix/")
async def get_confusion_matrix():
    matrix_path = "static/plots/confusion_matrix.png"
    if not os.path.exists(matrix_path):
        return {"error": "Confusion matrix not found. Train the model first."}
    return FileResponse(matrix_path)