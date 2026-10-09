import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱"
)


st.title("🌱 Plant Disease Detection using ResNet18")
st.write("Upload a plant leaf image to predict disease.")


# Classes
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper___Bacterial_spot",
    "Pepper___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___healthy"
]


device = "cpu"


# Load Model
@st.cache_resource
def load_model():

    try:

        model = models.resnet18(weights=None)

        model.fc = torch.nn.Linear(
            model.fc.in_features,
            len(class_names)
        )


        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "best_plant_disease_model.pth"
        )


        checkpoint = torch.load(
            model_path,
            map_location=device
        )


        model.load_state_dict(checkpoint)

        model.eval()

        return model


    except Exception as e:

        st.error(
            f"Model loading failed: {e}"
        )

        return None



model = load_model()



# Image preprocessing

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])



uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg","jpeg","png"]
)



if uploaded_file and model:


    image = Image.open(uploaded_file).convert("RGB")


    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )


    image_tensor = transform(image).unsqueeze(0)



    with torch.no_grad():

        output = model(image_tensor)


        probabilities = torch.nn.functional.softmax(
            output,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities,
            1
        )



    disease = class_names[predicted.item()]


    st.success(
        f"Prediction: {disease}"
    )


    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )