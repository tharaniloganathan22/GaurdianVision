import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)

# Set up directories
UPLOAD_FOLDER = 'static/uploads'
DETECTED_FOLDER = 'static/detected_faces'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

# Load trained model
MODEL_PATH = "face_mask_model.h5"
model = load_model(MODEL_PATH)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

def process_image(image_path):
    """Preprocess uploaded image and detect face mask."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128)) / 255.0
        face_reshaped = np.expand_dims(face_resized, axis=0)

        prediction = model.predict(face_reshaped)[0][0]
        confidence = round(prediction * 100, 2)

        label = "Mask" if prediction > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, f"{label} ({confidence}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save detected image
        detected_path = os.path.join(DETECTED_FOLDER, os.path.basename(image_path))
        cv2.imwrite(detected_path, image)

        results.append({"label": label, "confidence": confidence})

    return results, detected_path if results else None

@app.route("/", methods=["GET", "POST"])
def index():
    """Homepage: Upload image for mask detection."""
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded image
            results, detected_img = process_image(filepath)

            return render_template("result.html", results=results, detected_img=detected_img)
    
    return render_template("index.html")

@app.route("/evaluate")
def evaluate():
    """Display evaluation metrics with graphs."""
    y_true = []
    y_pred = []

    # Load evaluation images
    for file in os.listdir(DETECTED_FOLDER):
        label = 1 if "mask" in file.lower() else 0  # Extract true label
        image_path = os.path.join(DETECTED_FOLDER, file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        face_resized = cv2.resize(image, (128, 128)) / 255.0
        face_reshaped = np.expand_dims(face_resized, axis=0)
        prediction = model.predict(face_reshaped)[0][0]
        predicted_label = 1 if prediction > 0.5 else 0

        y_true.append(label)
        y_pred.append(predicted_label)

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Save classification report as an image
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Mask", "Mask"], yticklabels=["No Mask", "Mask"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("static/conf_matrix.png")
    
    return render_template("evaluation.html", report=report, conf_matrix="static/conf_matrix.png")

if __name__ == "__main__":
    app.run(debug=True)
