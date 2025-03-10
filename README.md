# GaurdianVision
Developed a real-time face mask detection web app using a CNN model for classification and OpenCV’s Haar Cascade for live face detection. Implemented a Flask-based interface for image uploads and webcam detection, with visualization using Seaborn for model evaluation. Achieved 94% accuracy in distinguishing masked and unmasked faces.

---

## 🚀 **Features**
✅ **Real-time detection**: Uses OpenCV to detect faces and classify them as "Mask" or "No Mask".  
✅ **Image Upload**: Users can upload images for face mask classification.  
✅ **Deep Learning Model**: Uses a pre-trained CNN for mask classification.  
✅ **Web Interface**: Built using **Flask**, **HTML**, **CSS**, and **JavaScript**.  
✅ **Evaluation Metrics**: Provides a **classification report** and **heatmap** for model performance visualization.  
✅ **Easy Deployment**: Works locally and can be deployed on **Heroku, AWS, or PythonAnywhere**.  

---


---

## 📊 **Dataset**
The model is trained on the **Face Mask Detection Dataset**, which contains **images of people with and without masks**.

- **Categories**:
  - `Mask` → Images of people wearing masks.
  - `No Mask` → Images of people without masks.
  
- **Data Augmentation** was applied to improve model performance.

---


---

## 🖼 **How It Works**
### **1️⃣ Image Upload**
- Users upload an image via the web UI.
- The system **detects faces using OpenCV**.
- The CNN model **classifies** each detected face as **Mask** or **No Mask**.

### **2️⃣ Real-Time Detection (Webcam)**
- The system captures frames using OpenCV.
- It detects faces in **real-time**.
- The trained CNN model classifies the detected faces.
- The system **draws a bounding box** around the face and labels it accordingly.

---

## 📈 **Model Training**
### **1️⃣ CNN Architecture**
The model is a **Convolutional Neural Network (CNN)** trained using TensorFlow/Keras.  
#### **Layers:**
- **Conv2D** (Feature Extraction)
- **MaxPooling2D** (Downsampling)
- **Flatten** (Convert to 1D)
- **Dense Layers** (Classification)

### **2️⃣ Evaluation Metrics**
The model was evaluated using:
- **Accuracy**: 94%
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **Loss & Accuracy Graphs**

---

## 📊 **Performance Visualization**
### **1️⃣ Confusion Matrix**
The confusion matrix shows how well the model differentiates between "Mask" and "No Mask."

### **2️⃣ Classification Report**
           precision    recall  f1-score   support

No Mask       0.94      0.94      0.94       766
   Mask       0.93      0.94      0.94       745

accuracy                           0.94      1511

### **3️⃣ Accuracy & Loss Graphs**
- The model converged well with minimal overfitting.
- Training accuracy reached **94%**.

---

Tech Stack
Backend: Flask
Frontend: HTML, CSS, JavaScript
Machine Learning: TensorFlow/Keras, Scikit-learn
Computer Vision: OpenCV
Visualization: Matplotlib, Seaborn


