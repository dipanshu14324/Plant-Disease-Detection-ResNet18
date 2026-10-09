# 🌱 Plant Disease Detection using ResNet18

An AI-based Plant Disease Detection System built using **Deep Learning (PyTorch + ResNet18)**.  
The application allows users to upload a plant leaf image and predicts the disease category with confidence score.

## 🚀 Demo

The project is deployed using Streamlit.

Features:
- Upload plant leaf image
- AI-based disease prediction
- Confidence score display
- Fast inference using trained ResNet18 model

---

## 🧠 Model Details

- Framework: PyTorch
- Architecture: ResNet18
- Task: Multi-class Image Classification
- Dataset: PlantVillage
- Number of Classes: 38

---

## 📂 Project Structure

📦 Requirements
Python 3.10+
PyTorch
Torchvision
Streamlit
Pillow
NumPy

📊 Model Limitations

The model is trained on the PlantVillage dataset containing 38 plant disease classes.

It performs best on supported plants such as:

Apple
Tomato
Potato
Corn
Grape
Peach

Images of unsupported plants (for example Mango or Neem) may produce incorrect predictions because those classes are not included in training data.