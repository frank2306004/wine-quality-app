import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Predictor (Binary: Low/High)")
st.write("Dự đoán chất lượng rượu vang dựa trên 11 đặc trưng hóa học + loại rượu (red/white).")

MODEL_PATH = Path("artifacts/model.joblib")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Không tìm thấy model tại artifacts/model.joblib. Hãy chạy `python train.py` để tạo model.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Nhập thông số")

wine_type = st.selectbox("Wine type", ["red", "white"])

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("fixed acidity", min_value=0.0, value=7.0, step=0.1)
    volatile_acidity = st.number_input("volatile acidity", min_value=0.0, value=0.3, step=0.01)
    citric_acid = st.number_input("citric acid", min_value=0.0, value=0.3, step=0.01)
    residual_sugar = st.number_input("residual sugar", min_value=0.0, value=3.0, step=0.1)
    chlorides = st.number_input("chlorides", min_value=0.0, value=0.05, step=0.001)
    free_sulfur_dioxide = st.number_input("free sulfur dioxide", min_value=0.0, value=30.0, step=1.0)

with col2:
    total_sulfur_dioxide = st.number_input("total sulfur dioxide", min_value=0.0, value=115.0, step=1.0)
    density = st.number_input("density", min_value=0.0, value=0.995, step=0.0001, format="%.4f")
    pH = st.number_input("pH", min_value=0.0, value=3.2, step=0.01)
    sulphates = st.number_input("sulphates", min_value=0.0, value=0.5, step=0.01)
    alcohol = st.number_input("alcohol", min_value=0.0, value=10.5, step=0.1)

row = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
    "wine_type": wine_type,
}
X = pd.DataFrame([row])

st.write("Dữ liệu đầu vào:")
st.dataframe(X, use_container_width=True)

if st.button("🔮 Predict"):
    # predict_proba nếu có
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= 0.5)
    else:
        pred = int(model.predict(X)[0])
        proba = None

    label = "HIGH (>=6)" if pred == 1 else "LOW (<=5)"
    st.success(f"Kết quả: **{label}**")

    if proba is not None:
        st.info(f"Xác suất HIGH: **{proba:.3f}**")
