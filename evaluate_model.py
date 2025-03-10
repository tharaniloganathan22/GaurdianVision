import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report

def load_trained_model(model_path):
    """Load the trained face mask detection model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")
    return load_model(model_path)

def process_image(image_path, img_size, model_input_shape):
    """Read, preprocess, and reshape an image for model inference."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Warning: Unable to read {image_path}. Skipping...")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    
    if len(model_input_shape) == 2:  # Expected flattened input
        image = image.flatten().reshape(1, -1)
    else:  # Expected CNN format
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image

def evaluate_model(model, detected_faces_path, img_size, model_input_shape):
    """Evaluate the model using detected face images."""
    if not os.path.exists(detected_faces_path):
        raise FileNotFoundError(f"❌ Directory '{detected_faces_path}' not found!")

    true_labels, predicted_labels = [], []
    image_files = [f for f in os.listdir(detected_faces_path) if f.endswith(".jpg")]
    
    if not image_files:
        raise ValueError("❌ No images found in detected_faces directory!")
    
    for file in image_files:
        file_path = os.path.join(detected_faces_path, file)
        image = process_image(file_path, img_size, model_input_shape)
        if image is None:
            continue

        # Predict using the model
        prediction = model.predict(image)[0][0]  # Assuming output is a single probability
        predicted_label = 1 if prediction > 0.5 else 0  # 1 = Mask, 0 = No Mask

        # Extract ground truth label from filename
        true_label = 1 if "mask" in file.lower() else 0

        # Append labels
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    
    print(f"✅ Processed {len(true_labels)} images in '{detected_faces_path}'")
    
    # Check for class imbalance
    if len(set(true_labels)) < 2 or len(set(predicted_labels)) < 2:
        print("⚠️ Warning: Only one class detected in true or predicted labels!")
        print("Ensure detected_faces folder has images of both 'mask' and 'no_mask'.")
    
    # Display classification report
    print("\n📊 Model Evaluation Report:")
    print(classification_report(true_labels, predicted_labels, labels=[0, 1], target_names=["No Mask", "Mask"]))

if __name__ == "__main__":
    model_path = "face_mask_model.h5"
    detected_faces_path = "detected_faces"

    model = load_trained_model(model_path)
    model_input_shape = model.input_shape
    IMG_SIZE = model_input_shape[1]  # Extract height & width

    evaluate_model(model, detected_faces_path, IMG_SIZE, model_input_shape)
