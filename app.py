# Import Required Libraries

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing import image

from tumor_info import tumor_information

# Page Configuration

st.set_page_config(
    page_title="Brain Tumor Detection using CNN",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS

st.markdown("""
<style>

.main-title{
    font-size:38px;
    font-weight:bold;
    color:#1E88E5;
    text-align:center;
}

.sub-title{
    font-size:18px;
    color:gray;
    text-align:center;
}

.result{
    font-size:24px;
    font-weight:bold;
    color:green;
}

</style>
""", unsafe_allow_html=True)

# Project Title

st.markdown(
    "<p class='main-title'>🧠 Brain Tumor Detection using CNN</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Upload an MRI Brain Scan Image to Predict the Tumor Type</p>",
    unsafe_allow_html=True
)

# Load Trained Model

model = tf.keras.models.load_model(
    "model/brain_tumor_model.keras"
)

# Class Labels

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Sidebar

st.sidebar.title("Project Information")

st.sidebar.info("""
Brain Tumor Detection using Deep Learning (CNN)

Dataset:
Brain MRI Images

Classes:
• Glioma
• Meningioma
• No Tumor
• Pituitary
""")

st.sidebar.metric(
    "Model Accuracy",
    "98%"
)

# Upload MRI Image

uploaded_image = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    # Open Uploaded Image

    img = Image.open(uploaded_image)

    img = img.convert("RGB")

    # Display Uploaded Image

    st.subheader("Uploaded MRI Image")

    st.image(
        img,
        caption="Uploaded Brain MRI Scan",
        use_container_width=True
    )

    # Prediction Button

    if st.button("Predict Tumor"):

        # Resize Image

        img = img.resize((224, 224))

        # Convert Image into Array

        img_array = image.img_to_array(img)

        # Add Batch Dimension

        img_array = np.expand_dims(img_array, axis=0)

        # Normalize Image

        img_array = img_array / 255.0

        # Model Prediction

        with st.spinner("Analyzing Brain MRI Image..."):         

            prediction = model.predict(img_array)
                                       
        # Get Predicted Class

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        # Confidence Score

        confidence = np.max(prediction) * 100


        # Display Result

        st.success("Prediction Completed Successfully")

        st.write(
            "Predicted Tumor Type:",
            predicted_class
        )

        st.write(
            "Confidence Score:",
            f"{confidence:.2f}%"
        )
        st.progress(float(confidence) / 100)

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.subheader("📊 Prediction Probability")

        prob_df = pd.DataFrame({
            "Tumor Type": class_names,
            "Probability": prediction[0]
        })

        st.bar_chart(
            prob_df.set_index("Tumor Type")
        )
                        
# Display Tumor Informationst.subheader("Prediction Probability")

if uploaded_image is not None:

    try:

        if 'predicted_class' in locals():

            st.subheader("🧠 Tumor Information")

            info = tumor_information.get(
                predicted_class,
                "Information not available"
            )

            st.info(info)

            st.subheader("💡 Health Tips")

            st.success("""
            ✔ Consult a Neurologist

            ✔ Do not self-diagnose

            ✔ Follow your MRI report

            ✔ Maintain a healthy lifestyle

            ✔ This prediction is for educational purposes only.
            """)

    except Exception as e:

        st.warning("Tumor information could not be loaded")


# Additional Project Details

st.markdown("---")

st.subheader("📌 About This Project")

st.write("""
This project uses a Convolutional Neural Network (CNN)
to classify Brain MRI images into four categories:

• Glioma Tumor
• Meningioma Tumor
• No Tumor
• Pituitary Tumor

The model is trained using a Brain MRI Image Dataset.
""")

# Model Details

st.subheader("⚙️ Model Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "CNN"
    )

with col2:
    st.metric(
        "Image Size",
        "224 x 224"
    )

with col3:
    st.metric(
        "Classes",
        "4"
    )

# Footer

st.markdown("---")

st.caption(
    "🧠 Brain Tumor Detection System | Developed using Streamlit, TensorFlow, CNN and Python"
)