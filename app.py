import streamlit as st
import numpy as np
import joblib
import os
import gdown

# URL or ID of your Google Drive model
model_id = '1dgR94Ty-H7PZeuXlbSBCbSFY3nAnL_IT'  # Replace with your actual file ID
model_file = 'rf_model.pkl'

# Check if model is already downloaded
if not os.path.exists(model_file):
    gdown.download(f"https://drive.google.com/uc?id={model_id}", model_file, quiet=False)

# Now you can load the model as usual
model = joblib.load(model_file)

# Load scaler
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Fire Type Classification", layout="wide")

# Sidebar Information
with st.sidebar:
    st.title("📘 Project Overview")
    st.markdown("""
    **Deforestation Detection: Fire Type Classification**

    This project uses **MODIS satellite data** to classify different fire types 
    such as **deforestation fires, forest fires,** or **no fire** using machine learning.

    ---
    ### ⚙️ Model Used:
    - Random Forest Classifier
    - Scikit-learn for preprocessing

    ### 📊 Features:
    - Brightness  
    - Brightness T31  
    - Fire Radiative Power (FRP)  
    - Scan  
    - Track  
    - Confidence Level

    ### 🔖 Labels:
    - **0**: No Fire  
    - **2**: Deforestation Fire  
    - **3**: Forest Fire

    ---

    **Author**: [Ishika Sharma](https://www.linkedin.com/in/ishika-sharma/)  
    **GitHub**: [skyish21](https://github.com/skyish21)

    ---
    """)

# Title and Description
st.markdown("## Deforestation Detection")
st.markdown("## 🔥 Fire Type Classification using MODIS Data")

# Description
st.write("""
This Streamlit app predicts the type of fire observed using remote sensing data collected from the MODIS satellite system.  
It takes in six key features related to fire activity and classifies the instance into one of the following fire types:
- 🌳 **Deforestation fires**
- 🌲 **Forest fires**
- 🚫 **No fire**

The goal is to support environmental monitoring by detecting harmful fire activities early and accurately.
""")

# User input fields
st.subheader("🔍 Enter Feature Values")

# Brightness input
brightness = st.slider("Brightness", min_value=290.0, max_value=500.0, value=300.0, step=1.0)
with st.expander("ℹ️ What is Brightness?"):
    st.markdown("**Brightness** is the thermal radiation from the fire spot. Higher values indicate more intense fires.")

# Brightness T31 input
bright_t31 = st.slider("Brightness T31", min_value=280.0, max_value=400.0, value=290.0, step=1.0)
with st.expander("ℹ️ What is Brightness T31?"):
    st.markdown("**Brightness T31** is the brightness temperature from channel 31 (IR). Used to filter out land heat.")

# FRP input
frp = st.slider("Fire Radiative Power (FRP)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
with st.expander("ℹ️ What is FRP?"):
    st.markdown("**FRP** represents energy emitted by the fire. Higher FRP → more intense fire activity.")

# Scan input
scan = st.slider("Scan", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
with st.expander("ℹ️ What is Scan?"):
    st.markdown("**Scan** represents the size of the satellite swath. Can affect resolution and detection.")

track = st.number_input("Track", value=1.0)
with st.expander("ℹ️ What is Track?"):
    st.markdown("**Track** refers to the fraction of the satellite's orbit path where the fire was detected. "
                "Higher values often indicate detections near the center of the swath, which are more accurate.")
    
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])
with st.expander("ℹ️ What is Confidence?"):
    st.markdown("**Confidence** is the quality score assigned to a fire detection.\n"
                "- **Low**: less reliable detection\n"
                "- **Nominal**: moderate confidence\n"
                "- **High**: highly reliable detection")

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("🔎 Predict Fire Type"):
    pred = model.predict(scaled_input)[0]
    fire_types = {0: "No Fire", 2: "Deforestation Fire", 3: "Forest Fire"}
    st.success(f"🔥 Predicted Fire Type: **{fire_types.get(pred, 'Unknown')}**")

# Show India Map in Main Area
st.markdown("### 🗺️ Fire Map of India")
with open("india_map.html", "r") as f:
    st.components.v1.html(f.read(), height=400, scrolling=True)


