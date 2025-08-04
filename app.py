import streamlit as st
import numpy as np
import pandas as pd
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

st.set_page_config(page_title="Fire Type Classification", layout="centered")

# Sidebar Information
with st.sidebar:
    st.markdown("## 📚 Info Panel")
    st.markdown("---")

    st.markdown("📘 Project Overview")
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
                
    """)

    st.markdown("---")
    st.markdown("## 👩‍💻 About Me")
    st.markdown(
        """
        <div style='text-align: center;'>
            <a href='https://github.com/skyish21' target='_blank'>GitHub 🔗</a>&nbsp;|&nbsp;
            <a href='https://www.linkedin.com/in/ishika-sharma-79a67a326/' target='_blank'>💼 LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Title and Description
st.markdown("# Deforestation Detection ")
st.markdown("### Fire Type Classification using MODIS Data")
st.markdown("---")

# Description
st.write("""
This Streamlit app predicts the type of fire observed using remote sensing data collected from the MODIS satellite system.  
It takes in six key features related to fire activity and classifies the instance into one of the following fire types:
- 🌳 **Deforestation fires**
- 🌲 **Forest fires**
- 🚫 **No fire**

The goal is to support environmental monitoring by detecting harmful fire activities early and accurately.
""")

with st.expander("Fire Map of India (Click to Expand)"):
    st.markdown("""
    This map provides a **visual overview of fire occurrences across different regions in India**, as detected by MODIS satellite data.

    **What it shows:**
    - Geolocations of recorded fire events.
    - Distribution of different **fire types** (e.g., deforestation, forest fire, or no fire).
    - Hotspot regions where fires are more frequent.

    **Why it's useful:**
    - Enables **spatial understanding** of fire patterns.
    - Helps **forest departments and researchers** identify high-risk zones.
    - Aids in **early warning and response planning**.

    """)

    # Show India Map
    with open("india_map.html", "r") as f:
        st.components.v1.html(f.read(), height=400, scrolling=True)

st.markdown("---")
# User input fields
st.subheader("Enter Feature Values 🔍")
st.markdown("---")

# Brightness input
brightness = st.slider("Brightness", min_value=290.0, max_value=500.0, value=300.0, step=1.0)
with st.expander("ℹ️ What is Brightness?"):
    st.markdown("""
    **Brightness** is the thermal radiation emitted from the fire spot, measured by the satellite.

    **Higher values**:
    - More intense fire
    - Larger flame or higher temperature

    **Lower values**:
    - Small or no fire
    - False detection due to warm land
    """)

# Brightness T31 input
bright_t31 = st.slider("Brightness T31", min_value=280.0, max_value=400.0, value=290.0, step=1.0)
with st.expander("ℹ️ What is Brightness T31?"):
    st.markdown("""
    **Brightness T31** is the temperature measured from infrared channel 31 — used as a background reference.

    **Higher values**:
    - May indicate hot background (like dry soil or rocks)
    - Could reduce contrast between fire and background

    **Lower values**:
    - Easier fire detection when background is cooler
    """)

# FRP input
frp = st.slider("Fire Radiative Power (FRP)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
with st.expander("ℹ️ What is FRP?"):
    st.markdown("""
    **FRP** represents the amount of energy emitted by the fire.

    **Higher FRP**:
    - More energetic fire → likely a **forest or deforestation fire**  
    
    **Lower FRP**:
    - May be **no fire** or a small heat source
    """)

# Scan input
scan = st.slider("Scan", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
with st.expander("ℹ️ What is Scan?"):
    st.markdown("""
    **Scan** is the angular width of the satellite swath capturing the fire.
    
    **Higher values**:
    - Edge of the swath → potential distortion or missed fires
    
    **Lower values**:
    - Object closer to nadir (center view) → more accurate detection  
    """)

# Track input
track = st.number_input("", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
with st.expander("ℹ️ What is Track?"):
    st.markdown("""
    **Track** represents the satellite’s position across its orbital path at the time of detection.

    **Higher values**: 
    - Central, more reliable readings  
    
    **Lower values**: 
    - Near edge, slightly less accurate
    """)

# Confidence input
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])
with st.expander("ℹ️ What is Confidence?"):
    st.markdown("""
    **Confidence** indicates how sure the system is that a fire is present.

    **High**: 
    - 90–100% certainty — very likely a real fire  
    
    **Nominal**: 
    - Medium probability  
    
    **Low**: 
    - Possibly a false detection or noise
    """)


# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])

# Scale input using pre-trained scaler
scaled_input = scaler.transform(input_data)

st.markdown("---")

# Prediction
if st.button("Predict Fire Type 🔎 "):
    pred = model.predict(scaled_input)[0]

    fire_types = {0: "No Fire", 2: "Deforestation Fire", 3: "Forest Fire"}
    st.success(f" 🔥 Predicted Fire Type: **{fire_types.get(pred, 'Unknown')}**")









