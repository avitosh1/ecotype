"""
 app_ecotype.py — EcoType: Forest Cover Type Prediction
Streamlit Application — Fully Requirement-Compliant

Pages:
  1. Predict         — Single input (with labeled dropdowns) + Auto-predict
  2. Batch Predict   — Upload CSV, download results
  3. Visualizations  — All training plots with descriptions
  4. Model Report    — Comparison table, classification report, manifest
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="EcoType — Forest Cover Classification",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------
# CUSTOM CSS — clean, professional look
# -----------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stMetric { background: #f0f8f0; border-radius: 10px; padding: 0.5rem 1rem; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1a6b3c;
        border-left: 4px solid #1a6b3c; padding-left: 10px;
        margin: 1.5rem 0 0.75rem 0;
    }
    .desc-box {
        background: #f7faf7; border: 1px solid #c8e6c9;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.88rem; color: #2d5a27; margin-bottom: 0.5rem;
    }
    .result-box {
        background: #e8f5e9; border: 2px solid #4caf50;
        border-radius: 12px; padding: 1.25rem 1.5rem;
        font-size: 1.2rem; font-weight: 600; color: #1b5e20;
        text-align: center; margin-top: 1rem;
    }
    .warn-box {
        background: #fff8e1; border: 1px solid #ffc107;
        border-radius: 8px; padding: 0.6rem 1rem;
        font-size: 0.87rem; color: #7a5800;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
ARTIFACTS = Path("artifacts")

# -----------------------------------------------------------------------
# WILDERNESS & SOIL LABEL MAPS (human-readable dropdowns)
# -----------------------------------------------------------------------
WILDERNESS_LABELS = {
    0: "Not Designated",
    1: "Rawah Wilderness",
    2: "Neota Wilderness",
    3: "Comanche Peak Wilderness",
    4: "Cache la Poudre Wilderness"
}

COVER_TYPE_NAMES = {
    1: "Spruce / Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood / Willow",
    5: "Aspen",
    6: "Douglas Fir",
    7: "Krummholz"
}

# Soil type descriptions (40 types — abbreviated for UI)
SOIL_LABELS = {i: f"Soil Type {i}" for i in range(1, 41)}
SOIL_LABELS.update({
    1:  "Cathedral family — rock outcrop complex, extremely stony",
    2:  "Vanet — Ratake families complex, very stony",
    3:  "Haploborolis — rock outcrop complex, rubbly",
    4:  "Ratake family — rock outcrop complex, rubbly",
    5:  "Vanet family — rock outcrop complex, rubbly",
    10: "Bullwark — Catamount — rock outcrop complex, rubbly",
    29: "Moran family — Cryorthents — Leighcan family, bouldery",
    40: "Cryaquepts — Typic Cryaquepts"
})

# -----------------------------------------------------------------------
# SAFE LOAD
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    def _load(p):
        try:
            with open(str(p), "rb") as f:
                return joblib.load(f)
        except Exception:
            return None

    model          = _load(ARTIFACTS / "best_model.pkl")
    label_enc      = _load(ARTIFACTS / "label_encoder.pkl")
    feature_summary= _load(ARTIFACTS / "feature_summary.pkl")

    manifest = None
    mp = ARTIFACTS / "run_manifest.json"
    if mp.exists():
        with open(str(mp)) as f:
            manifest = json.load(f)

    comparison = None
    cp = ARTIFACTS / "model_comparison.csv"
    if cp.exists():
        comparison = pd.read_csv(str(cp))

    report_txt = None
    rp = ARTIFACTS / "classification_report.txt"
    if rp.exists():
        report_txt = rp.read_text()

    tuning = None
    tp = ARTIFACTS / "tuning_results.csv"
    if tp.exists():
        tuning = pd.read_csv(str(tp))

    return model, label_enc, feature_summary, manifest, comparison, report_txt, tuning

model, label_enc, feature_summary, manifest, comparison_df, report_txt, tuning_df = load_artifacts()

# -----------------------------------------------------------------------
# GUARD — artifacts must exist
# -----------------------------------------------------------------------
if model is None:
    st.error("❌ `best_model.pkl` not found in `artifacts/`. Run the training script first.")
    st.stop()
if feature_summary is None:
    st.error("❌ `feature_summary.pkl` not found. Run the training script first.")
    st.stop()

continuous_cols = feature_summary["continuous_cols"]
binary_cols     = feature_summary["binary_cols"]

# Separate wilderness / soil binary cols from plain 0/1 binary cols
WILDERNESS_COLS = [c for c in binary_cols if "Wilderness_Area" in c]
SOIL_COLS       = [c for c in binary_cols if "Soil_Type" in c]
OTHER_BINARY    = [c for c in binary_cols if c not in WILDERNESS_COLS + SOIL_COLS]

# -----------------------------------------------------------------------
# CONTINUOUS FEATURE METADATA (for sensible defaults & ranges)
# -----------------------------------------------------------------------
FEATURE_META = {
    "Elevation":                          {"min": 1859, "max": 3858, "default": 2800, "unit": "m"},
    "Aspect":                             {"min": 0,    "max": 360,  "default": 180,  "unit": "°"},
    "Slope":                              {"min": 0,    "max": 66,   "default": 15,   "unit": "°"},
    "Horizontal_Distance_To_Hydrology":   {"min": 0,    "max": 1397, "default": 250,  "unit": "m"},
    "Vertical_Distance_To_Hydrology":     {"min": -173, "max": 601,  "default": 30,   "unit": "m"},
    "Horizontal_Distance_To_Roadways":    {"min": 0,    "max": 7117, "default": 1500, "unit": "m"},
    "Hillshade_9am":                      {"min": 0,    "max": 254,  "default": 200,  "unit": ""},
    "Hillshade_Noon":                     {"min": 0,    "max": 254,  "default": 220,  "unit": ""},
    "Hillshade_3pm":                      {"min": 0,    "max": 254,  "default": 140,  "unit": ""},
    "Horizontal_Distance_To_Fire_Points": {"min": 0,    "max": 7173, "default": 1500, "unit": "m"},
    "Hillshade_Mean":                     {"min": 0,    "max": 254,  "default": 186,  "unit": ""},
    "Hydrology_Euclidean":                {"min": 0,    "max": 1500, "default": 250,  "unit": "m"},
    "Elevation_Slope_Ratio":              {"min": 20,   "max": 3858, "default": 150,  "unit": ""},
}

def get_meta(col, key, fallback):
    return FEATURE_META.get(col, {}).get(key, fallback)

# -----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/evergreen-tree.png", width=60)
    st.title("EcoType")
    st.caption("Forest Cover Type Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📁 Batch Predict", "📊 Visualizations", "📋 Model Report"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    if manifest:
        st.markdown("**Last Training Run**")
        st.caption(f"Best model: **{manifest.get('best_model','—')}**")
        st.caption(f"Test accuracy: **{manifest.get('test_accuracy','—')}**")
        st.caption(f"CV score: **{manifest.get('best_cv_score','—')}**")
    st.markdown("---")
    st.info("Place all files from `artifacts/` in the same folder as `app.py` before running.")


# =======================================================================
# PAGE 1 — PREDICT
# =======================================================================
if page == "🔮 Predict":
    st.title("🌲 EcoType — Forest Cover Type Prediction")
    st.markdown("Enter the geographic and environmental features below to predict the forest cover type.")

    tab1, tab2 = st.tabs(["✏️ Manual Input", "🎲 Auto-Predict (Random)"])

    # -------------------------------------------------------------------
    # TAB 1 — MANUAL SINGLE PREDICTION
    # -------------------------------------------------------------------
    with tab1:
        with st.form("single_input_form"):
            user_data = {}

            # ── Continuous features ─────────────────────────────────────
            st.markdown('<div class="section-header">Terrain & Distance Features</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            for i, col in enumerate(continuous_cols):
                target_col = c1 if i % 2 == 0 else c2
                mn  = get_meta(col, "min",     0.0)
                mx  = get_meta(col, "max",     5000.0)
                dfv = get_meta(col, "default", 0.0)
                unit = get_meta(col, "unit",   "")
                label = f"{col.replace('_', ' ')} ({unit})" if unit else col.replace("_", " ")
                user_data[col] = target_col.number_input(
                    label, min_value=float(mn), max_value=float(mx),
                    value=float(dfv), step=1.0
                )

            # ── Wilderness Area dropdown ─────────────────────────────────
            if WILDERNESS_COLS:
                st.markdown('<div class="section-header">Wilderness Area</div>', unsafe_allow_html=True)
                wild_choice = st.selectbox(
                    "Select Wilderness Area",
                    options=list(WILDERNESS_LABELS.keys()),
                    format_func=lambda x: WILDERNESS_LABELS[x]
                )
                for col in WILDERNESS_COLS:
                    # extract index from name e.g. Wilderness_Area_1 → 1
                    try:
                        idx = int(col.split("_")[-1])
                    except ValueError:
                        idx = -1
                    user_data[col] = 1 if idx == wild_choice else 0

            # ── Soil Type dropdown ───────────────────────────────────────
            if SOIL_COLS:
                st.markdown('<div class="section-header">Soil Type</div>', unsafe_allow_html=True)
                soil_nums = []
                for col in SOIL_COLS:
                    try:
                        soil_nums.append(int(col.split("_")[-1]))
                    except ValueError:
                        soil_nums.append(-1)

                soil_choice = st.selectbox(
                    "Select Soil Type",
                    options=soil_nums if soil_nums else list(range(1, 41)),
                    format_func=lambda x: f"Type {x} — {SOIL_LABELS.get(x, '')}"
                )
                for col, num in zip(SOIL_COLS, soil_nums):
                    user_data[col] = 1 if num == soil_choice else 0

            # ── Other binary features ────────────────────────────────────
            if OTHER_BINARY:
                st.markdown('<div class="section-header">Other Binary Features</div>', unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                for i, col in enumerate(OTHER_BINARY):
                    target_col = b1 if i % 2 == 0 else b2
                    user_data[col] = target_col.selectbox(
                        col.replace("_", " "), options=[0, 1],
                        format_func=lambda x: "Yes (1)" if x else "No (0)"
                    )

            submitted = st.form_submit_button("🔍 Predict Cover Type", use_container_width=True)

        if submitted:
            df_input = pd.DataFrame([user_data])
            try:
                raw_pred = model.predict(df_input)[0]
                if label_enc is not None:
                    display_pred = label_enc.inverse_transform([raw_pred])[0]
                else:
                    display_pred = raw_pred
                cover_name = str(display_pred)
                st.markdown(
                    f'<div class="result-box">🌳 Predicted Cover Type: {display_pred} — {cover_name}</div>',
                    unsafe_allow_html=True
                )
                # Show input summary
                with st.expander("View input values used for prediction"):
                    st.dataframe(pd.DataFrame([user_data]).T.rename(columns={0: "Value"}))
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # -------------------------------------------------------------------
    # TAB 2 — AUTO PREDICT
    # -------------------------------------------------------------------
    with tab2:
        st.markdown("Generates a realistic random observation and predicts its cover type.")
        st.markdown('<div class="desc-box">Values are randomly sampled within realistic ranges for each feature based on the Covertype dataset distribution.</div>', unsafe_allow_html=True)

        n_samples = st.slider("Number of random samples to generate", 1, 10, 3)

        if st.button("🎲 Generate & Predict", use_container_width=True):
            rows = []
            for _ in range(n_samples):
                auto_data = {}
                for col in continuous_cols:
                    mn  = get_meta(col, "min",     0.0)
                    mx  = get_meta(col, "max",     1500.0)
                    auto_data[col] = float(np.random.uniform(mn, mx))

                # Wilderness — pick exactly one
                if WILDERNESS_COLS:
                    chosen = np.random.choice(WILDERNESS_COLS)
                    for col in WILDERNESS_COLS:
                        auto_data[col] = 1 if col == chosen else 0

                # Soil — pick exactly one
                if SOIL_COLS:
                    chosen = np.random.choice(SOIL_COLS)
                    for col in SOIL_COLS:
                        auto_data[col] = 1 if col == chosen else 0

                for col in OTHER_BINARY:
                    auto_data[col] = np.random.randint(0, 2)

                rows.append(auto_data)

            auto_df = pd.DataFrame(rows)

            try:
                preds = model.predict(auto_df)
                if label_enc is not None:
                    preds = label_enc.inverse_transform(preds)

                auto_df["Predicted_Cover_Type"] = preds
                auto_df["Cover_Type_Name"] = [
                    str(p) for p in preds
                ]

                st.success(f"✔ {n_samples} prediction(s) generated!")
                st.dataframe(auto_df[["Predicted_Cover_Type", "Cover_Type_Name"] +
                                     continuous_cols[:5]].style.highlight_max(
                    subset=continuous_cols[:5], color="#c8e6c9"), use_container_width=True)

                with st.expander("View full generated data"):
                    st.dataframe(auto_df, use_container_width=True)

            except Exception as e:
                st.error(f"Auto Prediction Error: {e}")


# =======================================================================
# PAGE 2 — BATCH PREDICT
# =======================================================================
elif page == "📁 Batch Predict":
    st.title("📁 Batch CSV Prediction")
    st.markdown("Upload a CSV file with the same feature columns as the training data.")

    st.markdown('<div class="desc-box">The CSV must contain all feature columns. The Cover_Type column is optional — if present it will be used for comparison. Download the template below to get started.</div>', unsafe_allow_html=True)

    # Template download
    required_cols = continuous_cols + binary_cols
    template_df = pd.DataFrame(columns=required_cols)
    st.download_button(
        "📥 Download CSV Template",
        template_df.to_csv(index=False).encode("utf-8"),
        "ecotype_template.csv", "text/csv"
    )

    st.markdown("---")
    file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            input_df = df[required_cols].copy()

            col_run, col_info = st.columns([1, 2])
            run = col_run.button("🔍 Run Batch Prediction", use_container_width=True)

            if run:
                try:
                    with st.spinner("Running predictions..."):
                        preds = model.predict(input_df)
                        if label_enc is not None:
                            preds_named = label_enc.inverse_transform(preds)
                        else:
                            preds_named = preds

                    df["Predicted_Cover_Type"] = preds_named
                    df["Cover_Type_Name"] = [
                        str(p) for p in preds_named
                    ]

                    st.success(f"✔ {len(df)} predictions complete!")

                    # Distribution of predictions
                    pred_counts = pd.Series(preds_named).value_counts().reset_index()
                    pred_counts.columns = ["Cover_Type", "Count"]
                    st.subheader("Prediction Distribution")
                    st.bar_chart(pred_counts.set_index("Cover_Type"))

                    st.subheader("Results Preview")
                    st.dataframe(df[["Predicted_Cover_Type", "Cover_Type_Name"] +
                                    continuous_cols[:4]].head(20), use_container_width=True)

                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Full Predictions CSV",
                        csv_out,
                        "ecotype_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Prediction Error: {e}")


# =======================================================================
# PAGE 3 — VISUALIZATIONS
# =======================================================================
elif page == "📊 Visualizations":
    st.title("📊 Training Visualizations")
    st.markdown("All plots generated during model training, with descriptions.")

    # Define all plots with detailed descriptions
    VIS_CATALOG = [
        {
            "file":    "class_distribution.png",
            "title":   "Class Distribution — Cover Type",
            "section": "EDA",
            "desc":    """Shows how many samples belong to each of the 7 forest cover types in the dataset.
                        A balanced distribution is ideal for classification. Imbalanced classes (where some types
                        have far fewer samples) can bias the model — this plot helps identify whether SMOTE
                        oversampling was necessary."""
        },
        {
            "file":    "histograms.png",
            "title":   "Univariate Histograms — Continuous Features",
            "section": "EDA",
            "desc":    """Frequency distribution of each continuous feature across the entire dataset.
                        Helps identify skewness (long tails), bimodal distributions, and the natural range of each
                        variable. Features with high skewness (|skew| ≥ 0.75) were corrected using log1p
                        transformation in Task 3."""
        },
        {
            "file":    "boxplots.png",
            "title":   "Boxplots by Cover Type (Top 6 Features)",
            "section": "EDA",
            "desc":    """Boxplots compare the spread and median of the top 6 continuous features across
                        the 7 cover type classes. Features where the boxes clearly separate between classes
                        (e.g., Elevation) are strong predictors. Outliers are shown as individual dots beyond
                        the whiskers — these were clipped using the IQR method in Task 3."""
        },
        {
            "file":    "correlation_matrix.png",
            "title":   "Correlation Matrix — Continuous Features",
            "section": "EDA",
            "desc":    """Pairwise Pearson correlation between all continuous features. Values close to +1 (dark red)
                        indicate strong positive correlation; values close to -1 (dark blue) indicate inverse
                        correlation. Highly correlated features carry redundant information — this informs
                        feature selection and the decision to drop low-variance features."""
        },
        {
            "file":    "feature_importance_plot.png",
            "title":   "Top 15 Feature Importances — Random Forest",
            "section": "Feature Selection",
            "desc":    """Feature importance scores from a Random Forest fitted on the preprocessed training data
                        (Task 7). Importance is measured by the average reduction in impurity (Gini) contributed
                        by each feature across all trees. Features with higher scores are more useful for
                        classification. Features below the mean threshold were dropped via SelectFromModel."""
        },
        {
            "file":    "model_comparison.png",
            "title":   "Model Comparison — CV Accuracy",
            "section": "Model Evaluation",
            "desc":    """Horizontal bar chart comparing the mean cross-validation (CV=3, StratifiedKFold) accuracy
                        of all 5 trained classifiers: Random Forest, Decision Tree, Logistic Regression, KNN,
                        and XGBoost. Error bars show the standard deviation across folds. The best-performing
                        model (highlighted in dark green) was selected for hyperparameter tuning."""
        },
        {
            "file":    "confusion_matrix.png",
            "title":   "Confusion Matrix — Best Model (Test Set)",
            "section": "Model Evaluation",
            "desc":    """The confusion matrix shows the number of correct and incorrect predictions for each
                        cover type class on the held-out test set (20% of data). Diagonal cells (dark blue)
                        are correct predictions. Off-diagonal cells are misclassifications — the row is the
                        true class and the column is the predicted class. High off-diagonal values for a
                        particular pair indicate which classes the model confuses most often."""
        },
    ]

    # Group by section
    sections = {}
    for item in VIS_CATALOG:
        sections.setdefault(item["section"], []).append(item)

    for section_name, items in sections.items():
        st.markdown(f'<div class="section-header">{section_name}</div>', unsafe_allow_html=True)
        for item in items:
            path = ARTIFACTS / item["file"]
            if path.exists():
                with st.expander(f"📈 {item['title']}", expanded=True):
                    desc_col, _ = st.columns([3, 0.01])
                    with desc_col:
                        st.markdown(
                            f'<div class="desc-box">{item["desc"]}</div>',
                            unsafe_allow_html=True
                        )
                    st.image(str(path), use_column_width=True)
            else:
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>{item["file"]}</b> not found in artifacts/. '
                    f'Re-run the training script to generate it.</div>',
                    unsafe_allow_html=True
                )
        st.markdown("")


# =======================================================================
# PAGE 4 — MODEL REPORT
# =======================================================================
elif page == "📋 Model Report":
    st.title("📋 Model Report & Evaluation")

    # ── Run manifest ────────────────────────────────────────────────────
    if manifest:
        st.subheader("Training Run Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Model",    manifest.get("best_model", "—"))
        m2.metric("Test Accuracy", f"{manifest.get('test_accuracy', 0):.4f}")
        m3.metric("Best CV Score", f"{manifest.get('best_cv_score', 0):.4f}")
        m4.metric("Random State",  manifest.get("random_state", 42))

        with st.expander("Best Hyperparameters"):
            st.code(manifest.get("best_params", "—"), language="python")

        with st.expander("Features Used"):
            feat_df = pd.DataFrame({
                "Feature": manifest.get("features_used", []),
                "Type": ["Derived" if f in manifest.get("derived_features", [])
                         else "Continuous / Binary"
                         for f in manifest.get("features_used", [])]
            })
            st.dataframe(feat_df, use_container_width=True)

        with st.expander("Skewness-Corrected Features (log1p applied)"):
            sf = manifest.get("skewed_features", [])
            if sf:
                st.write(sf)
            else:
                st.write("None")

    st.markdown("---")

    # ── Model comparison table ───────────────────────────────────────────
    if comparison_df is not None:
        st.subheader("Model Comparison — Cross-Validation Results")
        st.markdown('<div class="desc-box">All 5 models were evaluated using StratifiedKFold (k=3) cross-validation '
                    'on the training set. SMOTE oversampling was applied inside each fold to prevent data leakage. '
                    'The model with the highest mean CV accuracy was selected for hyperparameter tuning.</div>',
                    unsafe_allow_html=True)

        styled = comparison_df.style.background_gradient(
            subset=["CV_Accuracy_Mean"], cmap="Greens"
        ).format({"CV_Accuracy_Mean": "{:.4f}", "CV_Accuracy_Std": "±{:.4f}"})
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("model_comparison.csv not found in artifacts/.")

    st.markdown("---")

    # ── Classification report ────────────────────────────────────────────
    if report_txt:
        st.subheader("Classification Report — Test Set")
        st.markdown('<div class="desc-box">Precision, Recall, and F1-score per cover type class on the held-out '
                    'test set (20% of data). Macro avg treats all classes equally; weighted avg accounts for '
                    'class frequency. A high F1 for all classes indicates the model generalises well across '
                    'all forest cover types.</div>', unsafe_allow_html=True)
        st.code(report_txt, language="text")
    else:
        st.warning("classification_report.txt not found in artifacts/.")

    st.markdown("---")

    # ── Hyperparameter tuning results ───────────────────────────────────
    if tuning_df is not None:
        st.subheader("Hyperparameter Tuning Results (Top 10 Trials)")
        st.markdown('<div class="desc-box">Results from RandomizedSearchCV (n_iter=10, CV=3) on the best model. '
                    'Trials are sorted by rank — lower rank = better. The top-ranked parameter set was used '
                    'to produce the final model.</div>', unsafe_allow_html=True)
        show_cols = [c for c in ["params", "mean_test_score", "std_test_score", "rank_test_score"]
                     if c in tuning_df.columns]
        st.dataframe(
            tuning_df[show_cols].head(10).style.format(
                {"mean_test_score": "{:.4f}", "std_test_score": "±{:.4f}"}
            ).background_gradient(subset=["mean_test_score"], cmap="Blues"),
            use_container_width=True
        )
    else:
        st.warning("tuning_results.csv not found in artifacts/.")

    # ── Cover type legend ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Cover Type Reference")
    legend_df = pd.DataFrame(
        [(k, v) for k, v in COVER_TYPE_NAMES.items()],
        columns=["Cover_Type_ID", "Cover_Type_Name"]
    )
    st.dataframe(legend_df, use_container_width=True, hide_index=True)
