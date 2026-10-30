
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from detector import detect_devices


# Create folders for input and output images
SAMPLES = Path("samples")
OUTPUTS = Path("outputs")

SAMPLES.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


# Configure the web application
st.set_page_config(
    page_title="Computer Lab Device Counter",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ Computer Lab Device Counter")

st.write(
    "Upload a computer laboratory image to "
    "automatically detect and count devices."
)

st.info(
    "This prototype uses a pretrained AI model. "
    "Some devices may be missed or misclassified."
)


# Image upload section
uploaded_file = st.file_uploader(
    "Upload a computer lab image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect and Count Devices"):

        with st.spinner("Detecting devices..."):

            image_path = SAMPLES / "uploaded_image.jpg"
            image.save(image_path)

            annotated, counts = detect_devices(
                image_path,
                confidence=confidence
            )

            output_path = OUTPUTS / "annotated_result.jpg"

            cv2.imwrite(str(output_path), annotated)

        # Convert OpenCV BGR image to RGB
        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Detection Results")

        left, right = st.columns(2)

        with left:
            st.image(
                annotated_rgb,
                caption="Detected Devices",
                use_container_width=True
            )

        with right:
            st.subheader("Device Counts")

            if counts:
                for device, count in sorted(counts.items()):
                    st.metric(device, count)

                st.metric(
                    "Total Detected Devices",
                    sum(counts.values())
                )
            else:
                st.warning(
                    "No supported devices detected. "
                    "Try a clearer image or adjust the "
                    "confidence threshold."
                )

        # Download the annotated result
        success, encoded = cv2.imencode(
            ".jpg",
            annotated
        )

        if success:
            st.download_button(
                label="Download Annotated Image",
                data=encoded.tobytes(),
                file_name="lab_device_count.jpg",
                mime="image/jpeg"
            )
