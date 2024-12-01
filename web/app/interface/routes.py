from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, make_response, send_file
from datetime import datetime, timedelta
import requests
import pstats
import os
import uuid 


interface = Blueprint('interface', __name__)

FASTAPI_URL = "http://127.0.0.1:8000"


UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@interface.route('/')
@interface.route('/home')
def home():
    return render_template('index.html')


@interface.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the uploaded image
        image = request.files.get('image')

        if not image:
            flash('Please upload an image.', 'danger')
            return redirect(url_for('interface.predict'))

        try:
            # Save the uploaded image locally in the static/uploads folder
            unique_filename = f"{uuid.uuid4()}_{image.filename}"
            saved_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            image.save(saved_path)

            # Send image to FastAPI for prediction
            files = {'image': (image.filename, open(saved_path, 'rb'), image.mimetype)}
            response = requests.post(f"{FASTAPI_URL}/predict", files=files)
            response.raise_for_status()  # Raise an error for bad responses
            prediction_data = response.json()

            # Extract prediction results
            label = prediction_data.get('label', 'Unknown')
            confidence = prediction_data.get('confidence', 0.0)

            return render_template(
                'predict.html',
                prediction=label,
                confidence=f"{confidence:.2f}%",
                uploaded_image_url=url_for('static', filename=f"uploads/{unique_filename}")
            )

        except requests.exceptions.RequestException as e:
            flash(f"Error connecting to prediction service: {e}", 'danger')
            return redirect(url_for('interface.predict'))

    # Render the form if GET request
    return render_template('predict.html', prediction=None)


@interface.route('/model')
def model():
    return render_template('model.html')