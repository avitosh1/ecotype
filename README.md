# 🌲 EcoType: Forest Cover Type Prediction Using Machine Learning

EcoType is an end-to-end Machine Learning project that predicts the **forest cover type** of a geographical location using cartographic and environmental variables such as elevation, slope, soil type, hillshade, and wilderness area.

This project includes:

✅ Data Cleaning & Preprocessing
✅ Exploratory Data Analysis (EDA)
✅ Feature Engineering
✅ Multi-Class Classification Models
✅ Hyperparameter Tuning
✅ Model Evaluation
✅ Streamlit Web Application Deployment

---

# 📌 Project Objective

To build a machine learning model capable of accurately classifying forest cover type into one of **7 forest categories** based on terrain and environmental attributes.

This can help in:

* Forest Resource Management
* Environmental Monitoring
* Land Use Planning
* Wildfire Risk Analysis
* Ecological Research

---

# 📊 Dataset Information

* **Dataset Name:** Forest Cover Type Dataset
* **Rows:** 145,891
* **Columns:** 13
* **Target Variable:** `Cover_Type`
* **Classes:** 7 Forest Cover Types

### Input Features:

* Elevation
* Aspect
* Slope
* Horizontal Distance To Hydrology
* Vertical Distance To Hydrology
* Horizontal Distance To Roadways
* Hillshade 9am
* Hillshade Noon
* Hillshade 3pm
* Horizontal Distance To Fire Points
* Wilderness Area
* Soil Type

---

# 🌳 Target Classes

| ID | Cover Type          |
| -- | ------------------- |
| 1  | Spruce / Fir        |
| 2  | Lodgepole Pine      |
| 3  | Ponderosa Pine      |
| 4  | Cottonwood / Willow |
| 5  | Aspen               |
| 6  | Douglas Fir         |
| 7  | Krummholz           |

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Joblib
* Streamlit

---

# ⚙️ Machine Learning Workflow

## 1️⃣ Data Preprocessing

* Duplicate removal
* Missing value handling using imputation
* Outlier treatment using IQR clipping
* Skewness correction using log1p transformation

## 2️⃣ Feature Engineering

Created additional derived features:

* Hillshade_Mean
* Hydrology_Euclidean
* Elevation_Slope_Ratio

## 3️⃣ Class Imbalance Handling

Used **SMOTE** oversampling technique on training data.

## 4️⃣ Feature Selection

Used Random Forest feature importance + SelectFromModel.

## 5️⃣ Models Trained

* Random Forest
* Decision Tree
* Logistic Regression
* K-Nearest Neighbors (KNN)
* XGBoost

## 6️⃣ Hyperparameter Tuning

Used **RandomizedSearchCV** on best-performing model.

---

# 📈 Evaluation Metrics

* Accuracy Score
* Cross Validation Score
* Confusion Matrix
* Classification Report
* Feature Importance

---

# 🏆 Best Model Performance

*(Update with your final score)*

* **Best Model:** XGBoost / Random Forest
* **Cross Validation Accuracy:** XX.XX%
* **Test Accuracy:** XX.XX%

---

# 💻 Streamlit Application Features

## 🔮 Predict

Manual input of environmental features for single prediction.

## 🎲 Auto Predict

Generate realistic random samples and predict cover type.

## 📁 Batch Predict

Upload CSV file and download predictions.

## 📊 Visualizations

View all training charts and model insights.

## 📋 Model Report

Compare models, tuning results, and final metrics.

---

# 📂 Project Structure

```bash
EcoType_Project/
│── app.py
│── train_ecotype_final.py
│── covtype.csv
│── requirements.txt
│── README.md
│── artifacts/
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── preprocessor.pkl
│   ├── model_comparison.csv
│   ├── tuning_results.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── run_manifest.json
```

---

# ▶️ How to Run the Project

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Train Model

```bash
python train_ecotype_final.py
```

## 3️⃣ Launch Streamlit App

```bash
streamlit run app.py
```

---

# 📷 Sample Screens

* Prediction Dashboard
* Model Comparison Charts
* Batch Prediction Upload
* Confusion Matrix

(Add screenshots here if needed)

---

# 🔥 Key Highlights

✅ End-to-End ML Project
✅ Real-World Environmental Use Case
✅ Professional Streamlit Dashboard
✅ Multiple Models Compared
✅ Hyperparameter Tuning Included
✅ Deployment Ready

---

# 🚀 Future Improvements

* Cloud Deployment
* Real-time GIS Map Integration
* API Integration
* Deep Learning Models
* Satellite Image Classification

---

# 👨‍💻 Author

**Avitosh Sood**

---

# 📜 License

This project is for educational and portfolio purposes.

---
