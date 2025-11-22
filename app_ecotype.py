import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="EcoType — Forest Cover Classification",
    layout="wide"
)

st.title("🌲 EcoType — Forest Cover Type Prediction")
st.markdown("""
This app uses the trained machine learning model to predict the **forest cover type**
based on geographic and environmental features.
""")

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "best_model.pkl"
LABEL_PATH = ARTIFACTS / "label_encoder.pkl"
FEATURE_SUMMARY = ARTIFACTS / "feature_summary.pkl"

# -------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------
def load(path):
    try:
        return joblib.load(path)
    except:
        return None

model = load(MODEL_PATH)
label_enc = load(LABEL_PATH)
feature_summary = load(FEATURE_SUMMARY)

if model is None:
    st.error("❌ Could not load best_model.pkl. Run training script first and ensure artifacts exist.")
    st.stop()

if feature_summary is None:
    st.error("❌ feature_summary.pkl is missing. It is required for the UI.")
    st.stop()

continuous_cols = feature_summary["continuous_cols"]
binary_cols = feature_summary["binary_cols"]

st.sidebar.header("⚙️ Prediction Mode")
mode = st.sidebar.selectbox("Select Mode", ["Single Input", "Batch CSV"])

# -------------------------------------------------------
# INPUT FORM — SINGLE PREDICTION
# -------------------------------------------------------
if mode == "Single Input":
    st.header("🔹 Single Observation Input")

    with st.form("single_input"):
        user_data = {}

        st.subheader("Continuous Features")
        col1, col2 = st.columns(2)

        for i, c in enumerate(continuous_cols):
            col = col1 if i % 2 == 0 else col2
            val = col.number_input(c, value=0.0, format="%f")
            user_data[c] = val

        st.subheader("Binary Features")
        for c in binary_cols:
            user_data[c] = st.selectbox(c, [0, 1])

        submitted = st.form_submit_button("Predict Cover Type")

    if submitted:
        df_input = pd.DataFrame([user_data])

        try:
            pred = model.predict(df_input)[0]
            if label_enc is not None:
                pred = label_enc.inverse_transform([pred])[0]

            st.success(f"🌳 **Predicted Forest Cover Type: {pred}**")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # -------------------------------------------------------
    # AUTO-PREDICT RANDOM INPUT
    # -------------------------------------------------------
    st.subheader("✨ Auto-Predict (Generate Random Data Automatically)")

    if st.button("Generate Random Values & Predict"):
        auto_data = {}

        # Random realistic continuous values for forest data
        for c in continuous_cols:
            auto_data[c] = float(np.random.uniform(0, 1500))

        # Random binary flags
        for c in binary_cols:
            auto_data[c] = np.random.randint(0, 2)

        auto_df = pd.DataFrame([auto_data])

        st.write("🔹 **Generated Random Input:**")
        st.dataframe(auto_df)

        try:
            auto_pred = model.predict(auto_df)[0]

            if label_enc is not None:
                auto_pred = label_enc.inverse_transform([auto_pred])[0]

            st.success(f"🌟 **Auto-Predicted Cover Type: {auto_pred}**")

        except Exception as e:
            st.error(f"Auto Prediction Error: {e}")

# -------------------------------------------------------
# BATCH CSV PREDICTION
# -------------------------------------------------------
else:
    st.header("📁 Batch CSV Prediction")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Uploaded Dataset Preview:")
        st.dataframe(df.head())

        required_cols = continuous_cols + binary_cols
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            input_df = df[required_cols]

            try:
                preds = model.predict(input_df)

                if label_enc is not None:
                    preds = label_enc.inverse_transform(preds)

                df["Predicted_Cover_Type"] = preds
                st.success("✔ Predictions generated!")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions",
                    csv,
                    "ecotype_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# -------------------------------------------------------
# VISUALIZATIONS SECTION
# -------------------------------------------------------
st.header("📊 Visualizations from Model Training")

vis_files = {
    "Class Distribution": "class_distribution.png",
    "Correlation Matrix": "correlation_matrix.png",
    "Feature Importance (Top 15)": "feature_importance_plot.png",
    "Confusion Matrix": "confusion_matrix.png",
}

for title, filename in vis_files.items():
    file_path = ARTIFACTS / filename
    if file_path.exists():
        st.subheader(title)
        st.image(str(file_path), use_column_width=True)
    else:
        st.warning(f"⚠️ {filename} not found in artifacts/")

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("Ensure all artifacts are present in the artifacts/ folder before running the app.")
