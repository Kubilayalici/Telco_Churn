import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Telco Customer Churn Predictor")
st.caption("Model pipeline: ColumnTransformer + Best Classifier (GridSearched)")

MODEL_PATH = Path("best_model.pkl")
DATA_PATH = Path("notebook/data/Telco-Customer-Churn.csv")

@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_dataset():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # minimal cleanup for UI choices
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
        df["SeniorCitizen"] = df["SeniorCitizen"].astype('object')
        return df
    return None


model = load_model()
df = load_dataset()

if model is None:
    st.warning("Model bulunamadı. Lütfen önce `python notebook/model_trainer.py` komutuyla modeli eğitip `best_model.pkl` dosyasını oluşturun.")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Girdi Formu")
    if df is None:
        st.error("Veri seti bulunamadı: notebook/data/Telco-Customer-Churn.csv")
        st.stop()

    # Hedefi düş
    if 'Churn' in df.columns:
        X = df.drop(columns=['Churn'])
    else:
        X = df.copy()

    # Dinamik form: kategorikler için selectbox, sayısallar için number_input
    inputs = {}
    for col in X.columns:
        series = X[col]
        if series.dtype == 'O':
            opts = sorted(series.dropna().astype(str).unique().tolist())
            default = opts[0] if opts else ''
            inputs[col] = st.selectbox(col, options=opts, index=0 if default in opts else 0)
        else:
            val = float(series.dropna().median() if series.dropna().size > 0 else 0.0)
            inputs[col] = st.number_input(col, value=val)

    predict_btn = st.button("Tahmin Et")

with col2:
    st.subheader("Sonuç")
    if predict_btn and model is not None:
        row = pd.DataFrame([inputs])
        pred = model.predict(row)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = float(model.predict_proba(row)[:,1][0])
            except Exception:
                proba = None
        st.metric("Churn", "Evet" if int(pred)==1 else "Hayır")
        if proba is not None:
            st.metric("Churn Olasılığı", f"{proba:.3f}")

st.divider()
st.caption("Not: Model ve ön işleme pipeline'ı best_model.pkl içinde saklıdır.")

