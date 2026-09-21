import pickle
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 13 * 13, output_shape),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.convblock2(self.convblock1(inputs)))


MODEL_PATH = Path(__file__).with_name("model_LogR.sav")
CLASS_NAMES = (
    "Adenocarcinoma",
    "Large cell carcinoma",
    "Normal",
    "Squamous cell carcinoma",
)
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


@st.cache_resource
def load_model() -> nn.Module:
    setattr(sys.modules["__main__"], "TinyVGG", TinyVGG)
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    model.eval()
    return model


st.set_page_config(page_title="CT Scan Classifier", page_icon="🔬")
st.title("Chest CT Scan Classifier")
st.caption("Research demonstration only. This tool must not be used for clinical decisions.")

uploaded_file = st.file_uploader(
    "Upload one CT scan image",
    type=("png", "jpg", "jpeg"),
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model = load_model()
        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(image_tensor), dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        st.subheader(f"Prediction: {CLASS_NAMES[predicted_index]}")
        st.metric("Model confidence", f"{probabilities[predicted_index].item():.1%}")
        st.bar_chart(
            {CLASS_NAMES[index]: float(probabilities[index]) for index in range(len(CLASS_NAMES))}
        )
    except Exception as error:
        st.error(f"The model could not be loaded or run: {error}")
