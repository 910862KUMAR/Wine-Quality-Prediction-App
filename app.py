import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(
    page_title="🍷 AI Wine Quality Prediction",
    page_icon="🍇",
    layout="centered"
)

# --- Custom CSS for Background & Styling ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #2E0249, #570A57, #A91079);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        h1, h2, h3 {
            color: #F9F9F9;
            text-align: center;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            color: #EAEAEA;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #D91656;
            color: white;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-size: 18px;
            font-weight: bold;
            border: 1px solid #FFB6C1;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF6B81;
            color: black;
            transform: scale(1.02);
        }
        .stNumberInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        footer {
            text-align: center;
            color: #DDDDDD;
            font-size: 14px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("🍇 AI Wine Quality Prediction App")
st.markdown("<p class='subtitle'>Predict wine quality using Machine Learning and real chemical data 🍷</p>", unsafe_allow_html=True)

# --- Banner Image (small & centered, no message if missing) ---
image_path = os.path.join(os.path.dirname(__file__), "wine.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, width=400)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Load Model and Scaler ---
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Input Section ---
st.markdown("<h3>🧪 Enter Wine Chemical Properties:</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 4.0, 15.0, 7.4)
    volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)
    chlorides = st.number_input("Chlorides", 0.01, 0.20, 0.076)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1.0, 70.0, 11.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6.0, 300.0, 34.0)
    density = st.number_input("Density", 0.9900, 1.0040, 0.9978)
    pH = st.number_input("pH", 2.5, 4.5, 3.51)
    sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

# --- Prediction Button ---
if st.button("🔍 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>🍷 Prediction Result:</h3>", unsafe_allow_html=True)

    if prediction == 1:
        st.success(f"✅ This wine is **Good Quality!** (Confidence: {probability*100:.2f}%)")
        st.balloons()
        st.markdown("<p style='text-align:center; font-size:18px; color:#00FFB2;'>Cheers! This is a high-quality wine 🍾</p>", unsafe_allow_html=True)
    else:
        st.error(f"❌ This wine is **Low Quality.** (Confidence: {(1-probability)*100:.2f}%)")
        st.markdown("<p style='text-align:center; font-size:18px; color:#FFC3A0;'>May need some improvement in composition 🍇</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<footer>
    © 2025 | Developed by <b>Kumar GK</b> 🍇 | Powered by Machine Learning & Streamlit  
    <br> #AI #DataScience #WineQualityPrediction #Streamlit
</footer>
""", unsafe_allow_html=True)
