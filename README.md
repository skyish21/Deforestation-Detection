# Deforestation Detection: Fire Type Classification using MODIS Data 🔥 

This project predicts fire types (e.g., deforestation fire, forest fire, or no fire) based on satellite data from the MODIS system using various machine learning models — primarily a **Random Forest Classifier**.

---

## 📌 Project Objective

To build an intelligent system that classifies fire incidents using MODIS satellite data to help with **deforestation monitoring**, **wildfire tracking**, and **environmental safety**.

---

## 🧪 Features Used

| Feature Name        | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Brightness          | Thermal radiation from fire location. Higher value = intense fire.          |
| Brightness T31      | Infrared background temperature. Helps distinguish real fires.              |
| FRP (Fire Radiative Power) | Energy released by fire. Higher = more intense fire activity.        |
| Scan                | Width of satellite scan at detection. Impacts resolution.                   |
| Track               | Satellite’s orbital position at detection. Affects detection accuracy.      |
| Confidence Level    | Fire detection certainty: low, nominal, or high.                            |

---

## 🏷️ Labels

| Label Value | Class Name          |
|-------------|---------------------|
| 0           | No Fire             |
| 2           | Deforestation Fire  |
| 3           | Forest Fire         |

---

## 🤖 Models Tested and Accuracy

| Model                     | Accuracy Score     |
|---------------------------|--------------------|
| Logistic Regression       | 0.586              |
| Decision Tree             | 0.951              |
| **Random Forest**         | **0.979**          |
| K-Nearest Neighbors       | 0.933              |
| Gradient Boosting         | 0.613              |
| LightGBM                  | 0.929              |

📌 **Note**: Tree-based models like Gradient Boosting and LightGBM were also tested to explore improvements in performance. However, **Random Forest outperformed all** in accuracy and generalization.

---

## 🧠 Model Details

- **Best Model**: Random Forest Classifier
- **Preprocessing**: StandardScaler on 6 selected features
- **Trained On**: Cleaned and encoded MODIS data
- **Saved Model File**: [rf_model.pkl](https://drive.google.com/file/d/1dgR94Ty-H7PZeuXlbSBCbSFY3nAnL_IT/view?usp=drive_link)(Google Drive)

---

## 🖥️ Streamlit Web App

https://deforestation-detection-vhhqfw4fdfxr5bphwzcxjs.streamlit.app/


The Streamlit app allows users to:

- Input six key features using sliders and dropdowns
- Understand what each feature means and how it impacts prediction
- View a **Fire Map of India** showing historical fire points
- Predict whether input values indicate **No Fire**, **Deforestation Fire**, or **Forest Fire**

📍 India Map is embedded using a `india_map.html` file for interactive display.

---

## 📂 Folder Structure

```bash
.
├── app.py                    # Streamlit app
├── scaler.pkl                # Scaler used to normalize input features
├── rf_model.pkl              # Trained Random Forest model
├── india_map.html            # Interactive HTML map
├── README.md                 # You are here!
├── requirements.txt          # Python dependencies
└── data/                     # Raw and processed MODIS data (optional)

```

---

## 🔧 Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/skyish21/Deforestation-Detection.git
cd Deforestation-Detection
```

2. Install Required Packages
- Make sure you have Python ≥3.8 and pip installed. Then run:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit App
- You can use gdown in app.py to load the model directly from Google Drive.
- This will launch the web interface in your browser where you can try out fire classification using MODIS data
  
```bash
streamlit run app.py
```


