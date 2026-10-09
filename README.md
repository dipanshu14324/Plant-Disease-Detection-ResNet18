# 🌱 Plant Disease Detection using ResNet18

An AI-based Plant Disease Detection System built using **Deep Learning, PyTorch, and ResNet18**. Users can upload a plant leaf image to receive a predicted disease category and a confidence score.

## 🚀 Live Demo

**Try the app:** [Plant Disease Detection — Streamlit](https://plant-disease-detection-resnet18.streamlit.app/)

## 💻 GitHub Repository

**Source code:** [Plant-Disease-Detection-ResNet18](https://github.com/dipanshu14324/Plant-Disease-Detection-ResNet18)

## ✨ Features

- Upload a plant leaf image (JPG, JPEG, or PNG)
- Predict among 38 PlantVillage classes
- Display the model's prediction and confidence score
- Simple web interface built with Streamlit
- Image preprocessing and inference using PyTorch

## 🧠 Model Details

- **Framework:** PyTorch
- **Architecture:** ResNet18
- **Task:** Multi-class image classification
- **Dataset:** PlantVillage
- **Number of classes:** 38
- **Input size:** 224 × 224 pixels

## ⚙️ How It Works

1. Upload a leaf image in the web app.
2. The image is converted to RGB, resized to 224 × 224, and normalized.
3. The trained ResNet18 model processes the image.
4. The app displays the predicted class and confidence score.

## 📂 Project Structure

```text
Plant-Disease-Detection-ResNet18/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── best_plant_disease_model.pth
└── notebooks/
    └── Plant Diseases Detection .ipynb
```

## 🛠️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/dipanshu14324/Plant-Disease-Detection-ResNet18.git
   cd Plant-Disease-Detection-ResNet18
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## 📊 Dataset and Limitations

The model is trained on the PlantVillage dataset and recognizes 38 supported plant/disease classes. It performs best on images similar to the training data. Plants not represented in those classes (for example, mango or neem) may receive an incorrect prediction. The confidence score is the model's confidence in its selected class; it is **not a guarantee** that the prediction is correct.

## 🔮 Future Improvements

- Add more plant species and disease categories
- Evaluate the model with precision, recall, F1-score, and a confusion matrix
- Add a way to flag unsupported or unfamiliar plant images
- Improve performance on real-world images with varied lighting and backgrounds

## 👨‍💻 Author

**Dipanshu Shukla**

---

If you find this project useful, please ⭐ the repository.
