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

# ─── Generator definition (same as your training script) ───────────────────────
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

# ─── Main UI ───────────────────────────────────────────────────────────────────
st.title("🖋️ Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained GAN.")

# pick a digit
digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

# header always visible
st.header(f"Generated images of digit {digit}")

# sample 5 images immediately
with torch.no_grad():
    z      = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    imgs   = generator(z, labels).cpu()            # (5,1,28,28)
    imgs   = (imgs * 0.5 + 0.5).numpy()            # scale to [0,1]

# display them side-by-side
cols = st.columns(5)
for img_arr, col in zip(imgs, cols):
    arr = (img_arr.squeeze() * 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    col.image(pil, use_container_width=True)
