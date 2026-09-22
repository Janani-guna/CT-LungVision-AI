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


st.set_page_config(page_title="CT Scan Classifier", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #071b2a 0%, #0d2436 40%, #153a4f 100%);
            color: #ecf3ff;
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #ecf3ff;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .subtitle {
            color: #bcd5ef;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .glass {
            background: rgba(13, 35, 52, 0.8);
            border: 1px solid rgba(160, 200, 255, 0.22);
            border-radius: 18px;
            padding: 1.2rem 1.1rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
        }
        .prediction-card {
            background: rgba(8, 32, 45, 0.9);
            border: 1px solid rgba(131, 214, 255, 0.28);
            border-radius: 18px;
            padding: 1.2rem;
        }
        .stFileUploader > div {
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(160,200,255,0.35);
            border-radius: 12px;
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Chest CT Scan Classifier")
st.markdown(
    '<div class="subtitle">AI-assisted lung cancer screening demo for CT image classification.</div>',
    unsafe_allow_html=True,
)
st.caption("Research demonstration only. This tool must not be used for clinical decisions.")

with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload one CT scan image",
        type=("png", "jpg", "jpeg"),
        help="Supported formats: PNG, JPG, JPEG",
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    left_col, right_col = st.columns([1.25, 1])

    with left_col:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        try:
            model = load_model()
            image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
            with torch.inference_mode():
                probabilities = torch.softmax(model(image_tensor), dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())

            st.subheader("Prediction")
            st.markdown(
                f"<h3 style='color:#9fe7ff; margin-top:0; margin-bottom:0.4rem;'>{CLASS_NAMES[predicted_index]}</h3>",
                unsafe_allow_html=True,
            )
            st.metric("Model confidence", f"{probabilities[predicted_index].item():.1%}")
            st.bar_chart(
                {CLASS_NAMES[index]: float(probabilities[index]) for index in range(len(CLASS_NAMES))}
            )
        except Exception as error:
            st.error(f"The model could not be loaded or run: {error}")
        st.markdown('</div>', unsafe_allow_html=True)
