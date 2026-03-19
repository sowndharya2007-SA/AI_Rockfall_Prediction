import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
model = load_model("model/rockfall_lstm_model.h5")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(layout="wide")

st.title("⛰ AI Rockfall Monitoring System")

# -----------------------------
# TABS (MAIN FIX)
# -----------------------------
tab1, tab2 = st.tabs(["📡 Live Monitoring", "📊 Analysis & Metrics"])

# =========================================================
# 📡 TAB 1 — LIVE MONITORING
# =========================================================
with tab1:

    st.sidebar.header("System Control")

    auto_run = st.sidebar.checkbox("Enable Live Monitoring")

    st.sidebar.subheader("Sensor Controls")

    vibration = st.sidebar.slider("Vibration", 0.0, 1.0, 0.3)
    tilt = st.sidebar.slider("Tilt", 0, 10, 2)
    crack_width = st.sidebar.slider("Crack Width", 0.0, 1.0, 0.2)
    rainfall = st.sidebar.slider("Rainfall", 0, 200, 50)
    temperature = st.sidebar.slider("Temperature", 0, 50, 25)

    # -----------------------------
    # SESSION STATE
    # -----------------------------
    if "sensor_history" not in st.session_state:
        st.session_state.sensor_history = pd.DataFrame({
            "vibration": [],
            "tilt": [],
            "crack_width": [],
            "rainfall": [],
            "temperature": []
        }).astype(float)

    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []

    col1, col2 = st.columns(2)

    # -----------------------------
    # LIVE SYSTEM
    # -----------------------------
    if auto_run:

        for i in range(50):

            vib = float(vibration + np.random.uniform(-0.05,0.05))
            tl = float(tilt + np.random.uniform(-1,1))
            crack = float(crack_width + np.random.uniform(-0.05,0.05))
            rain = float(rainfall + np.random.randint(-10,10))
            temp = float(temperature + np.random.randint(-2,2))

            new_row = pd.DataFrame([{
                "vibration": vib,
                "tilt": tl,
                "crack_width": crack,
                "rainfall": rain,
                "temperature": temp
            }])

            st.session_state.sensor_history = pd.concat(
                [st.session_state.sensor_history, new_row],
                ignore_index=True
            )

            # -----------------------------
            # PREDICTION
            # -----------------------------
            data = np.array([[vib, tl, crack, rain, temp]])
            data_scaled = scaler.transform(data)
            data_lstm = data_scaled.reshape((1,1,5))

            prob = model.predict(data_lstm)[0][0]
            st.session_state.risk_history.append(prob)

            # -----------------------------
            # GRAPH DISPLAY
            # -----------------------------
            with col1:
                st.subheader("📊 Sensor Data")
                st.line_chart(st.session_state.sensor_history.astype(float))

            with col2:
                st.subheader("🧠 Risk Trend")
                st.line_chart(st.session_state.risk_history)

            # -----------------------------
            # ALERT SYSTEM
            # -----------------------------
            if prob > 0.7:
                st.error("🚨 CRITICAL ALERT!")
            elif prob > 0.5:
                st.warning("⚠ HIGH RISK")
            elif prob > 0.3:
                st.warning("⚠ MEDIUM RISK")
            else:
                st.success("SAFE")

            time.sleep(1)

    else:
        st.info("Enable Live Monitoring from sidebar")

    # -----------------------------
    # CURRENT STATUS
    # -----------------------------
    st.header("📊 Current Status")

    if st.session_state.risk_history:
        latest = st.session_state.risk_history[-1]
        st.metric("Current Risk", f"{latest*100:.2f}%")

    # -----------------------------
    # EXPORT
    # -----------------------------
    st.header("📂 Export Data")

    if st.button("Download Sensor Data"):
        st.download_button(
            label="Download CSV",
            data=st.session_state.sensor_history.to_csv(index=False),
            file_name="sensor_data.csv"
        )

# =========================================================
# 📊 TAB 2 — ANALYSIS (IMPORTANT FOR MARKS)
# =========================================================
with tab2:

    st.header("📊 Model Evaluation")

    try:
        cm = np.load("model/confusion_matrix.npy")

        st.write("Confusion Matrix:")
        st.write(cm)

        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
        st.write(f"Accuracy: {accuracy*100:.2f}%")

    except:
        st.warning("Run training first")

    # -----------------------------
    st.header("📂 Dataset")

    df = pd.read_csv("data/rockfall_dataset.csv")
    st.dataframe(df)

    # -----------------------------
    st.header("📊 Correlation Heatmap")

    corr = df.corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    fig.colorbar(cax)

    st.pyplot(fig)