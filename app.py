# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Digit Image Generator",
    page_icon="✍️",
    layout="wide",
)

# ─── Generator definition ───────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        lbl = self.label_emb(labels)
        x = torch.cat([noise, lbl], dim=1)
        return self.model(x)

# ─── Cache the generator so it only loads once ─────────────────────────────────
@st.cache_resource
def load_generator(path: str = "generator.pth") -> Generator:
    gen = Generator()
    state = torch.load(path, map_location="cpu")
    gen.load_state_dict(state)
    gen.eval()
    return gen

generator = load_generator()

# ─── Main page content ─────────────────────────────────────────────────────────
st.title("🖋️ Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-style images using your trained GAN.")

# ─── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Settings")
digit = st.sidebar.selectbox("Choose a digit (0–9):", list(range(10)))
if st.sidebar.button("Generate Images"):
    st.header(f"Generated images of digit {digit}")

    # sample 5 noise vectors and generate
    with torch.no_grad():
        z      = torch.randn(5, 100)
        labels = torch.full((5,), digit, dtype=torch.long)
        imgs   = generator(z, labels).cpu()
        imgs   = (imgs * 0.5 + 0.5).numpy()  # scale to [0,1]

    # display side by side
    cols = st.columns(5)
    for img_arr, col in zip(imgs, cols):
        arr = (img_arr.squeeze() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
        col.image(pil, use_container_width=True)
