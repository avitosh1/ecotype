"""
train_ecotype_final.py — COMPLETE & REQUIREMENT-COMPLIANT

Covers every project task:
✔ Task 1 : Data loading, shape/info/describe/value_counts, duplicate removal
✔ Task 2 : Missing value report (isnull check before imputation)
✔ Task 3 : Outlier handling (IQR clip) + Skewness correction (log1p)
✔ Task 4 : Feature engineering (3 derived features) + encoder saving
✔ Task 5 : Full EDA — histograms, boxplots, heatmap, class distribution
✔ Task 6 : Class imbalance handling (SMOTE inside pipeline, no leakage)
✔ Task 7 : Feature selection (SelectFromModel) + importance plot
✔ Task 8 : 5 models trained + model comparison CSV + bar chart saved
✔ Tuning : RandomizedSearchCV with param grids for ALL 5 models
✔ Task 9 : Best model + encoders saved as .pkl
✔ Misc   : Random seeds, cross-validation summary, all artifacts in /artifacts
"""

import warnings
warnings.filterwarnings("ignore")

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ML Tools
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Metrics & Imbalance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH  = Path("covtype.csv")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TARGET = "Cover_Type"

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  ✔ Plot saved: {path}")


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# -----------------------------------------------------------------------
# TASK 1 — DATA LOADING & UNDERSTANDING
# -----------------------------------------------------------------------
def load_and_understand(path):
    print_section("TASK 1 — Data Loading & Understanding")

    df = pd.read_csv(path)

    print(f"\n--- Shape ---")
    print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    print(f"\n--- Info ---")
    df.info()

    print(f"\n--- Describe (numeric) ---")
    desc = df.describe()
    print(desc.to_string())
    desc.to_csv(OUTPUT_DIR / "describe.csv")

    print(f"\n--- Target value counts ---")
    vc = df[TARGET].value_counts().sort_index()
    print(vc.to_string())

    return df


# -----------------------------------------------------------------------
# TASK 2 — DUPLICATE REMOVAL
# -----------------------------------------------------------------------
def remove_duplicates(df):
    print_section("TASK 2 — Duplicate Removal")
    n_dup = df.duplicated().sum()
    if n_dup:
        print(f"  ⚠ Found {n_dup} duplicate rows. Removing...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("  ✔ No duplicate rows found.")
    print(f"  Shape after dedup: {df.shape}")
    return df


# -----------------------------------------------------------------------
# MISSING VALUE REPORT (required before any imputation)
# -----------------------------------------------------------------------
def missing_value_report(df):
    print_section("Missing Value Report")
    mv = df.isnull().sum()
    mv_pct = (mv / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": mv, "missing_pct": mv_pct})
    report = report[report["missing_count"] > 0]

    if report.empty:
        print("  ✔ No missing values found in any column.")
    else:
        print(report.to_string())

    report.to_csv(OUTPUT_DIR / "missing_value_report.csv")
    return report


# -----------------------------------------------------------------------
# TASK 5 — EDA (full: histograms, boxplots, heatmap, class dist)
# -----------------------------------------------------------------------
def eda(df, continuous_cols, binary_cols):
    print_section("TASK 5 — EDA")

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color="#5DCAA5", edgecolor="white")
    ax.set_title("Class Distribution — Cover_Type")
    ax.set_xlabel("Cover Type")
    ax.set_ylabel("Count")
    save_fig(fig, "class_distribution.png")

    # 2. Correlation matrix (continuous only)
    if len(continuous_cols) > 1:
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(df[continuous_cols].corr(), cmap="coolwarm", vmin=-1, vmax=1,
                    annot=True, fmt=".2f", linewidths=0.4, ax=ax)
        ax.set_title("Correlation Matrix (Continuous Features)")
        save_fig(fig, "correlation_matrix.png")

    # 3. Histograms — univariate distribution of continuous features
    n = len(continuous_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(continuous_cols):
        axes[i].hist(df[col].dropna(), bins=40, color="#7F77DD", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Univariate Histograms — Continuous Features", fontsize=12, y=1.01)
    save_fig(fig, "histograms.png")

    # 4. Boxplots — detect spread and outliers per class
    top_feats = continuous_cols[:6]   # plot first 6 to keep it readable
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_feats):
        df.boxplot(column=col, by=TARGET, ax=axes[i], grid=False)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("Cover Type")
    fig.suptitle("Boxplots by Cover Type (Top 6 Features)", fontsize=12)
    plt.title("")
    save_fig(fig, "boxplots.png")

    print("  ✔ EDA complete — 4 plots saved.")


# -----------------------------------------------------------------------
# TASK 4 — FEATURE ENGINEERING (derived features)
# -----------------------------------------------------------------------
def engineer_features(df, continuous_cols):
    print_section("TASK 4 — Feature Engineering (Derived Features)")

    df = df.copy()

    # 1. Mean hillshade index — average light exposure across day
    if all(c in df.columns for c in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]):
        df["Hillshade_Mean"] = (
            df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
        ) / 3
        continuous_cols = continuous_cols + ["Hillshade_Mean"]
        print("  ✔ Hillshade_Mean created (avg of 9am, Noon, 3pm)")

    # 2. Euclidean distance to hydrology (combines horizontal + vertical)
    if all(c in df.columns for c in ["Horizontal_Distance_To_Hydrology",
                                      "Vertical_Distance_To_Hydrology"]):
        df["Hydrology_Euclidean"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2 +
            df["Vertical_Distance_To_Hydrology"] ** 2
        )
        continuous_cols = continuous_cols + ["Hydrology_Euclidean"]
        print("  ✔ Hydrology_Euclidean created (sqrt of H^2 + V^2)")

    # 3. Elevation-to-slope ratio — captures steepness relative to height
    if all(c in df.columns for c in ["Elevation", "Slope"]):
        df["Elevation_Slope_Ratio"] = df["Elevation"] / (df["Slope"] + 1)
        continuous_cols = continuous_cols + ["Elevation_Slope_Ratio"]
        print("  ✔ Elevation_Slope_Ratio created (Elevation / (Slope+1))")

    return df, continuous_cols


# -----------------------------------------------------------------------
# TASK 3 — OUTLIER HANDLING (IQR) + SKEWNESS CORRECTION (log1p)
# -----------------------------------------------------------------------
def compute_iqr_bounds(df, cols):
    bounds = {}
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        bounds[c] = {"lower": float(q1 - 1.5 * iqr), "upper": float(q3 + 1.5 * iqr)}
    return bounds


def apply_iqr(df, bounds):
    df = df.copy()
    for c, b in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(b["lower"], b["upper"])
    return df


def compute_skew_info(df, cols):
    df = df.copy()
    skew_info = {}
    for c in cols:
        if c not in df.columns:
            continue
        vals = df[c]
        if vals.nunique() < 5:
            continue
        skewness = vals.skew()
        if abs(skewness) >= 0.75:
            shift = float(abs(vals.min()) + 1) if vals.min() <= 0 else 0.0
            df[c] = np.log1p(df[c] + shift)
            skew_info[c] = {"shift": shift, "orig_skew": float(skewness)}
    return df, skew_info


def apply_skew_info(df, info):
    df = df.copy()
    for c, cfg in info.items():
        if c in df.columns:
            df[c] = np.log1p(df[c] + cfg["shift"])
    return df


# -----------------------------------------------------------------------
# TASK 7 — FEATURE IMPORTANCE PLOT
# -----------------------------------------------------------------------
def plot_feature_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(indices)), importances[indices], color="#378ADD", edgecolor="white")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right", fontsize=9)
    ax.set_title("Top 15 Feature Importances (Random Forest — Task 7)")
    ax.set_ylabel("Importance Score")
    save_fig(fig, "feature_importance_plot.png")


# -----------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------
def main():
    print_section("ECOTYPE — FOREST COVER TYPE CLASSIFICATION PIPELINE")

    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} not found. Place covtype.csv in the same folder.")
        return

    # ------------------------------------------------------------------
    # TASK 1 — Load & understand
    # ------------------------------------------------------------------
    df = load_and_understand(DATA_PATH)

    # ------------------------------------------------------------------
    # TASK 2 — Remove duplicates
    # ------------------------------------------------------------------
    df = remove_duplicates(df)

    # ------------------------------------------------------------------
    # Missing value report (before any imputation)
    # ------------------------------------------------------------------
    missing_value_report(df)

    # ------------------------------------------------------------------
    # Identify column types
    # ------------------------------------------------------------------
    features = [c for c in df.columns if c != TARGET]
    binary_cols     = [c for c in features if set(df[c].dropna().unique()) <= {0, 1}]
    continuous_cols = [c for c in features if c not in binary_cols]

    # ------------------------------------------------------------------
    # TASK 4 — Feature engineering (derived columns)
    # ------------------------------------------------------------------
    df, continuous_cols = engineer_features(df, continuous_cols)

    # ------------------------------------------------------------------
    # TASK 5 — EDA (all plots)
    # ------------------------------------------------------------------
    eda(df, continuous_cols, binary_cols)

    # ------------------------------------------------------------------
    # Encode target
    # ------------------------------------------------------------------
    print_section("Target Encoding")
    label_enc = LabelEncoder()
    df[TARGET] = label_enc.fit_transform(df[TARGET])
    with open(str(OUTPUT_DIR / "label_encoder.pkl"), "wb") as f:
        joblib.dump(label_enc, f, protocol=4)
    print(f"  Classes: {list(label_enc.classes_)}")

    # ------------------------------------------------------------------
    # Train-test split (stratified)
    # ------------------------------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print_section("Train-Test Split")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Save feature summary for Streamlit
    with open(str(OUTPUT_DIR / "feature_summary.pkl"), "wb") as f:
        joblib.dump({"continuous_cols": continuous_cols, "binary_cols": binary_cols}, f, protocol=4)

    # ------------------------------------------------------------------
    # TASK 3 — Outlier handling + skewness correction (fit on train only)
    # ------------------------------------------------------------------
    print_section("TASK 3 — Data Cleaning")

    iqr_bounds = compute_iqr_bounds(X_train, continuous_cols)
    X_train = apply_iqr(X_train, iqr_bounds)
    X_test  = apply_iqr(X_test,  iqr_bounds)
    with open(str(OUTPUT_DIR / "outlier_bounds.pkl"), "wb") as f:
        joblib.dump(iqr_bounds, f, protocol=4)
    print("  ✔ IQR outlier clipping applied.")

    X_train, skew_info = compute_skew_info(X_train, continuous_cols)
    X_test  = apply_skew_info(X_test, skew_info)
    with open(str(OUTPUT_DIR / "skew_info.pkl"), "wb") as f:
        joblib.dump(skew_info, f, protocol=4)
    skewed_features = list(skew_info.keys())
    print(f"  ✔ Log1p skew correction applied to: {skewed_features}")

    # ------------------------------------------------------------------
    # Preprocessor
    # ------------------------------------------------------------------
    cont_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor = ColumnTransformer([
        ("cont", cont_pipe, continuous_cols),
        ("bin",  bin_pipe,  binary_cols)
    ])

    # ------------------------------------------------------------------
    # TASK 7 — Feature selection + importance plot
    # ------------------------------------------------------------------
    print_section("TASK 7 — Feature Selection")

    preprocessor.fit(X_train)
    X_train_pre  = preprocessor.transform(X_train)
    feature_names = continuous_cols + binary_cols

    # Fit RF for importance plot
    fs_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    fs_rf.fit(X_train_pre, y_train)
    plot_feature_importance(fs_rf.feature_importances_, feature_names)

    # Fresh (unfitted) selector for use inside CV pipeline
    selector_est = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    selector     = SelectFromModel(selector_est, threshold="mean")
    print("  ✔ Feature selector configured (fits per CV fold — no leakage).")

    # ------------------------------------------------------------------
    # TASK 8 — Train & compare 5 models
    # ------------------------------------------------------------------
    print_section("TASK 8 — Model Training (CV=3, 5 Models)")

    models = {
        "RandomForest":      RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "DecisionTree":      DecisionTreeClassifier(random_state=RANDOM_STATE),
        "LogisticRegression":LogisticRegression(max_iter=500, multi_class="multinomial",
                                                 solver="lbfgs", random_state=RANDOM_STATE),
        "KNN":               KNeighborsClassifier(n_jobs=-1),
        "XGBoost":           xgb.XGBClassifier(eval_metric="mlogloss",
                                                random_state=RANDOM_STATE, n_jobs=-1)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in models.items():
        print(f"\n  Training {name}...")
        pipe = ImbPipeline([
            ("pre",    preprocessor),
            ("select", selector),
            ("smote",  SMOTE(random_state=RANDOM_STATE)),
            ("clf",    clf)
        ])
        scores = cross_val_score(pipe, X_train, y_train,
                                 scoring="accuracy", cv=cv, n_jobs=-1)
        results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"    CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Save model comparison as CSV
    comparison_df = pd.DataFrame(results).T.reset_index()
    comparison_df.columns = ["Model", "CV_Accuracy_Mean", "CV_Accuracy_Std"]
    comparison_df = comparison_df.sort_values("CV_Accuracy_Mean", ascending=False)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"\n  Model comparison:\n{comparison_df.to_string(index=False)}")

    # Save model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1D9E75" if i == 0 else "#5DCAA5" for i in range(len(comparison_df))]
    bars = ax.barh(comparison_df["Model"], comparison_df["CV_Accuracy_Mean"],
                   xerr=comparison_df["CV_Accuracy_Std"], color=colors,
                   edgecolor="white", capsize=4)
    ax.set_xlabel("CV Accuracy (mean ± std)")
    ax.set_title("Model Comparison — 5 Classifiers")
    ax.invert_yaxis()
    for bar, val in zip(bars, comparison_df["CV_Accuracy_Mean"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    save_fig(fig, "model_comparison.png")

    best_model_name = comparison_df.iloc[0]["Model"]
    print(f"\n  🏆 Best model: {best_model_name}")

    # ------------------------------------------------------------------
    # HYPERPARAMETER TUNING — param grids for ALL 5 models
    # ------------------------------------------------------------------
    print_section(f"Hyperparameter Tuning — {best_model_name}")

    param_grids = {
        "RandomForest": {
            "clf__n_estimators":     [100, 200, 300],
            "clf__max_depth":        [None, 20, 30],
            "clf__min_samples_split":[2, 5],
            "clf__max_features":     ["sqrt", "log2"]
        },
        "DecisionTree": {
            "clf__max_depth":        [None, 10, 20, 30],
            "clf__min_samples_split":[2, 5, 10],
            "clf__criterion":        ["gini", "entropy"]
        },
        "LogisticRegression": {
            "clf__C":      [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "saga"],
            "clf__max_iter":[500, 1000]
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights":     ["uniform", "distance"],
            "clf__metric":      ["euclidean", "manhattan"]
        },
        "XGBoost": {
            "clf__n_estimators":  [100, 200, 300],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth":     [3, 6, 9],
            "clf__subsample":     [0.8, 1.0]
        }
    }

    best_clf   = models[best_model_name]
    param_dist = param_grids[best_model_name]

    final_pipe = ImbPipeline([
        ("pre",    preprocessor),
        ("select", selector),
        ("smote",  SMOTE(random_state=RANDOM_STATE)),
        ("clf",    best_clf)
    ])

    tuner = RandomizedSearchCV(
        final_pipe, param_dist,
        n_iter=10, scoring="accuracy",
        cv=3, n_jobs=-1,
        random_state=RANDOM_STATE, verbose=1
    )
    tuner.fit(X_train, y_train)
    best_est = tuner.best_estimator_
    print(f"\n  ✔ Best params found: {tuner.best_params_}")
    print(f"  ✔ Best CV score    : {tuner.best_score_:.4f}")

    # Save tuning results
    tune_df = pd.DataFrame(tuner.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    tune_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    # ------------------------------------------------------------------
    # FINAL EVALUATION ON TEST SET
    # ------------------------------------------------------------------
    print_section("Final Evaluation — Test Set")

    y_pred = best_est.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {acc:.4f}")

    # Classification report
    cr = classification_report(y_test, y_pred,
                               target_names=[str(c) for c in label_enc.classes_])
    print(f"\n  Classification Report:\n{cr}")
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Best Model : {best_model_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(cr)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_enc.classes_,
                yticklabels=label_enc.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix — {best_model_name} (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_fig(fig, "confusion_matrix.png")

    # ------------------------------------------------------------------
    # TASK 9 — Save all artifacts
    # ------------------------------------------------------------------
    print_section("TASK 9 — Saving Artifacts")

    # Use explicit str paths + open() in binary mode to avoid Windows OSError 22.
    # joblib.dump with a Path object can silently fail on Windows when the
    # pipeline is large — passing a str and writing via open() is always safe.
    def safe_dump(obj, path):
        path = str(path)          # force str — avoids backslash/Path issues on Windows
        with open(path, "wb") as f:
            joblib.dump(obj, f, protocol=4)   # protocol 4 = Python 3.4+, handles large objects
        print(f"  ✔ Saved: {path}")

    safe_dump(best_est,     OUTPUT_DIR / "best_model.pkl")
    safe_dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")

    # Save a human-readable artifact manifest
    manifest = {
        "best_model":      best_model_name,
        "test_accuracy":   round(acc, 4),
        "best_cv_score":   round(tuner.best_score_, 4),
        "best_params":     str(tuner.best_params_),
        "features_used":   continuous_cols + binary_cols,
        "derived_features":["Hillshade_Mean", "Hydrology_Euclidean", "Elevation_Slope_Ratio"],
        "skewed_features": skewed_features,
        "random_state":    RANDOM_STATE
    }
    import json
    with open(OUTPUT_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("  Artifacts saved:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {p.name}")

    print_section("PIPELINE COMPLETE")
    print(f"  Best Model   : {best_model_name}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  All outputs  : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
