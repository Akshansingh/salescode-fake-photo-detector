import subprocess
import tempfile
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Spot the Fake Photo", page_icon="📸")
st.title("📸 Spot the Fake Photo")
st.write("Live camera demo for detecting whether a photo is real or a recaptured screen/printout.")

image_file = st.camera_input("Take a photo using your camera")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Captured Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, quality=95)
        temp_path = temp_file.name

    try:
        output = subprocess.check_output(["python", "predict.py", temp_path], text=True).strip()
        score = float(output)
        st.metric("Recapture / Screen Score", f"{score:.4f}")
        if score >= 0.5:
            st.error("Likely PHOTO OF A SCREEN / RECAPTURE")
        else:
            st.success("Likely REAL PHOTO")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
