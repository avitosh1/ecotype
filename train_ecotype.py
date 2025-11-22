"""
train_ecotype_final.py — FINAL (FIXED & DEPLOYMENT SAFE)

This file includes:
✔ Task 2: Duplicate record removal
✔ Task 3: Outlier handling (IQR) & Skew correction (Log1p)
✔ Task 7: Feature Selection (SelectFromModel) & Importance Plotting
✔ Task 8: 5 Models (RF, DT, LR, KNN, XGB)
✔ Task 6: Class Imbalance Handling (SMOTE)
✔ Hyperparameter Tuning (RandomizedSearchCV)
✔ FIX: Removed prefit=True to prevent NotFittedError during cross_val_score
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import random
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

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH = Path("covtype.csv")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"✔ Saved plot: {path}")

def eda(df, target):
    """Generate minimal EDA artifacts required for the project report."""
    print("Running EDA...")
    
    # Save basic stats
    df.describe().to_csv(OUTPUT_DIR / "describe.csv")
    
    # Class Distribution
    fig, ax = plt.subplots(figsize=(6,4))
    df[target].value_counts().sort_index().plot(kind="bar", ax=ax, color='skyblue')
    ax.set_title("Class Distribution")
    save_fig(fig, "class_distribution.png")

    # Correlation Matrix (Numerical only)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        save_fig(fig, "correlation_matrix.png")

def compute_iqr_bounds(df, cols):
    bounds = {}
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        bounds[c] = {"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr}
    return bounds

def apply_iqr(df, bounds):
    df = df.copy()
    for c, b in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(b["lower"], b["upper"])
    return df

def compute_skew(df, cols):
    df = df.copy()
    skew_info = {}
    for c in cols:
        vals = df[c]
        if vals.nunique() < 5: continue
        skewness = vals.skew()
        # Threshold 0.75 for significant skew
        if abs(skewness) >= 0.75:
            shift = 0
            if vals.min() <= 0:
                shift = abs(vals.min()) + 1
            df[c] = np.log1p(df[c] + shift)
            skew_info[c] = {"shift": float(shift), "orig_skew": float(skewness)}
    return df, skew_info

def apply_skew_info(df, info):
    df = df.copy()
    for c, cfg in info.items():
        shift = cfg["shift"]
        df[c] = np.log1p(df[c] + shift)
    return df

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    print("--- STARTING ECOTYPE PIPELINE ---")
    
    # 1. LOAD DATA
    if not DATA_PATH.exists():
        print(f"❌ Error: {DATA_PATH} not found. Please place the CSV in the same folder.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Original Shape: {df.shape}")

    # 2. DUPLICATE REMOVAL (Task 2)
    # ---------------------------------------------------------------
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Found {duplicates} duplicate rows. Removing...")
        df.drop_duplicates(inplace=True)
        print(f"New Shape: {df.shape}")
    else:
        print("✔ No duplicates found.")

    target = "Cover_Type"
    
    # 3. EDA
    eda(df, target)

    # 4. PREPARE COLUMNS
    features = [c for c in df.columns if c != target]
    binary_cols = [c for c in features if set(df[c].unique()) <= {0,1}]
    continuous_cols = [c for c in features if c not in binary_cols]

    # Save summary for Streamlit app
    joblib.dump({"continuous_cols": continuous_cols, "binary_cols": binary_cols}, 
                OUTPUT_DIR / "feature_summary.pkl")

    # 5. ENCODE TARGET
    y = df[target]
    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y)
    joblib.dump(label_enc, OUTPUT_DIR / "label_encoder.pkl")
    df[target] = y_enc

    # 6. TRAIN-TEST SPLIT
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 7. DATA CLEANING (Task 3)
    # ---------------------------------------------------------------
    print("\n--- Cleaning Data (Outliers & Skewness) ---")
    
    # Outliers
    iqr_bounds = compute_iqr_bounds(X_train, continuous_cols)
    X_train = apply_iqr(X_train, iqr_bounds)
    X_test = apply_iqr(X_test, iqr_bounds)
    joblib.dump(iqr_bounds, OUTPUT_DIR / "outlier_bounds.pkl")
    
    # Skewness
    X_train, skew_info = compute_skew(X_train, continuous_cols)
    X_test = apply_skew_info(X_test, skew_info)
    joblib.dump(skew_info, OUTPUT_DIR / "skew_info.pkl")
    print("✔ Data cleaning complete.")

    # 8. PREPROCESSOR SETUP
    cont_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("cont", cont_pipe, continuous_cols),
        ("bin", bin_pipe, binary_cols)
    ])

    # 9. FEATURE SELECTION & PLOTTING (Task 7)
    # ---------------------------------------------------------------
    print("\n--- Feature Selection ---")
    
    # Fit preprocessor temporarily to get clean data for selection
    preprocessor.fit(X_train)
    X_train_pre = preprocessor.transform(X_train)
    
    # Get approximate feature names (Continuous first, then Binary)
    feature_names = continuous_cols + binary_cols 
    
    # Train Random Forest specifically for Feature Importance PLOTTING (Task requirement)
    # We keep this step to generate the required plot.
    fs_rf_plot = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    fs_rf_plot.fit(X_train_pre, y_train)
    
    # Save Plot
    importances = fs_rf_plot.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Top 15 Feature Importances (Task 7)")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    save_fig(fig, "feature_importance_plot.png")
    
    # --- CRITICAL FIX START ---
    # Create a FRESH Selector for the Pipeline
    # We cannot use prefit=True inside cross_val_score because the pipeline cloning process
    # creates unfitted copies of the estimator. We must allow the selector to fit within each fold.
    selector_est = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1) 
    selector = SelectFromModel(selector_est, threshold="mean")
    # --- CRITICAL FIX END ---
    
    print("✔ Feature selection configured (will fit per CV fold).")

    # 10. MODEL TRAINING (Task 8)
    # ---------------------------------------------------------------
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=400, multi_class="multinomial"),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE)
    }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    print("\n--- Training 5 Models (CV=3) ---")
    for name, clf in models.items():
        print(f"Training {name}...")
        
        # PIPELINE: Preprocess -> Select Features -> Balance (SMOTE) -> Classify
        pipe = ImbPipeline([
            ("pre", preprocessor),
            ("select", selector),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf)
        ])
        
        score = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1).mean()
        results[name] = score
        print(f"   Accuracy: {score:.4f}")

    best_model_name = max(results, key=results.get)
    print(f"\n🏆 Best Model Selected: {best_model_name}")
    best_clf = models[best_model_name]

    # 11. HYPERPARAMETER TUNING
    # ---------------------------------------------------------------
    print(f"\n--- Tuning {best_model_name} (RandomizedSearchCV) ---")
    
    # Define hyperparams based on the winner
    if best_model_name == "RandomForest":
        param_dist = {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [None, 20, 30],
            "clf__min_samples_split": [2, 5]
        }
    elif best_model_name == "XGBoost":
         param_dist = {
            "clf__n_estimators": [200, 300],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 6, 9]
        }
    else:
        # Generic fallback
        param_dist = {}

    final_pipe = ImbPipeline([
        ("pre", preprocessor),
        ("select", selector),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", best_clf)
    ])

    if param_dist:
        tuner = RandomizedSearchCV(
            final_pipe, param_dist, n_iter=5, scoring="accuracy", 
            cv=3, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
        )
        tuner.fit(X_train, y_train)
        best_est = tuner.best_estimator_
        print(f"✔ Best Params: {tuner.best_params_}")
    else:
        final_pipe.fit(X_train, y_train)
        best_est = final_pipe

    # 12. FINAL EVALUATION & ARTIFACTS
    # ---------------------------------------------------------------
    print("\n--- Final Evaluation on Test Set ---")
    y_pred = best_est.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Save Report
    cr = classification_report(y_test, y_pred)
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(cr)

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title(f"Confusion Matrix ({best_model_name})")
    save_fig(fig, "confusion_matrix.png")

    # Save Final Model
    joblib.dump(best_est, OUTPUT_DIR / "best_model.pkl")
    # Save preprocessor explicitly for debugging, though it's inside best_model.pkl pipeline too
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")

    print("\n🎉 ALL TASKS COMPLETE. Artifacts saved in '/artifacts/'")

if __name__ == "__main__":
    main()