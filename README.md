# Early Gastric Cancer Detection Project
## Project Overview

The Early Gastric Cancer Detection System leverages a machine learning pipeline to assist healthcare professionals in predicting whether endoscopic images indicate the presence of cancerous lesions. This project includes a web-based FastAPI backend for prediction and model training and is built with a focus on scalability, accuracy, and user accessibility.

## Features

- **Image Classification**: Predicts if an uploaded endoscopic image indicates gastric cancer.
- **Model Training**: Enables users to train or retrain models using custom datasets.
- **Interactive Visualizations**: Displays model performance metrics such as training loss, accuracy, and confusion matrices.
- **REST API**: Supports endpoint-based interaction for seamless integration with external services.
- **Scalable Deployment**: Dockerized infrastructure for cloud deployment, tested on Microsoft Azure.

## File Structure

```
Project_name/
├── README.md          # Documentation
├── notebook/          # Jupyter Notebooks for experimentation
│   └── refined_model_Daniel_Ndungu_Summative_Assignment.ipynb
├── src/               # Core functionality
│   ├── preprocessing.py  # Image preprocessing utilities
│   ├── model.py          # ML model definition and training logic
│   └── prediction.py     # Model inference and prediction
├── data/              # Dataset directory
│   ├── cancerous/             # Training images
│   └── non-cancerous/              # Test images
├── models/            # Trained model storage
│   ├── refined_2.keras   
│   └── vanilla_model.keras    
```

## Installation Instructions

### Prerequisites

- Python 3.10 or higher
- Virtual environment setup (venv or conda)
- Docker (optional for deployment)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mwangi-dan/early-gastric-cancer-detection_MLOP_Summative.git
cd early-gastric-cancer-detection_MLOP_Summative
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Start the FastAPI Server

Run the following command in the project directory:

```bash
uvicorn fast:app --reload
```

The server will start on [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Endpoints Overview

- **Prediction**: `/predict/`
    - Upload an endoscopic image and select a trained model to predict its class.
- **Model Training**: `/train-model/`
    - Upload a zipped folder with labeled images (cancerous, non-cancerous) to train a new model.
- **Model Listing**: `/list-models/`
    - Fetch all available models for prediction.

### 3. Run Locust for Load Testing (Optional)

Start Locust to simulate user traffic:

```bash
locust -f locustfile.py
```

Visit [http://localhost:8089](http://localhost:8089) to configure and run tests.

## Deployment

### Using Docker

Build the Docker image:

```bash
docker build -t fastapi-app .
```

Run the Docker container:

```bash
docker run -p 8000:8000 fastapi-app
```

Optionally, use Docker Compose for Flask-FastAPI integration:

```bash
docker-compose up
```

## Live Demo

Explore the live project deployed on Microsoft Azure: [Early Gastric Cancer Detection Live Site](https://gastriccancerapp.azurewebsites.net/)

## Video Demo

Click below to watch the full walkthrough of the system:
[Link to Youtube Video Live Demo](https://youtu.be/kIsQD86w0-o)

## Future Improvements

- **Optimization**: Improve training speed and reduce memory usage.
- **GPU Support**: Integrate GPU-based model training for faster performance.
- **Extended Functionality**: Include additional metrics and support for more types of medical imagery.