# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Enhancement App", layout="centered")
st.title("📸 Live Color Image Enhancement (Sharp Output)")

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("Enhancement Controls")

noise_strength = st.sidebar.slider(
    "Noise Reduction Strength",
    min_value=0,
    max_value=15,
    value=6
)

contrast_strength = st.sidebar.slider(
    "Contrast Enhancement (CLAHE)",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1
)

sharpness = st.sidebar.slider(
    "Sharpness Level",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1
)

# ================= IMAGE CAPTURE =================
img_file_buffer = st.camera_input("Capture a live photo")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image = np.array(image)

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    st.subheader("Original Image")
    st.image(image, channels="RGB")

    # ================= PROCESSING PIPELINE =================

    # 1️⃣ Noise Reduction (controlled)
    if noise_strength > 0:
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=noise_strength,
            hColor=noise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
    else:
        denoised = image.copy()

    # 2️⃣ Convert to LAB color space
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 3️⃣ CLAHE on L channel
    clahe = cv2.createCLAHE(
        clipLimit=contrast_strength,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4️⃣ Unsharp Masking (Sharpness Control)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharp = cv2.addWeighted(
        enhanced,
        sharpness,
        gaussian,
        -(sharpness - 1),
        0
    )

    # ================= DISPLAY =================

    st.subheader("Enhanced Color Image")
    st.image(sharp, channels="RGB")

    # ================= DOWNLOAD =================

    Image.fromarray(sharp).save("enhanced_color_image.png")

    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=open("enhanced_color_image.png", "rb"),
        file_name="enhanced_color_image.png",
        mime="image/png"
    )
