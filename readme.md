# Heart Disease Risk Prediction

## Overview
This project focuses on predicting the risk of heart disease using classical machine learning models on structured clinical data. The goal is **not** to build a flashy web product, but to demonstrate an end-to-end, defensible ML workflow suitable for healthcare decision-support systems.

The project covers data preprocessing, model training, model comparison, and a **local-only Streamlit application** for experimentation and demonstration.

---

## Problem Statement
Heart disease is one of the leading causes of mortality worldwide. Early risk prediction using routinely collected clinical features can assist healthcare professionals in identifying high-risk patients and prioritizing intervention.

This project formulates heart disease prediction as a **binary classification problem**, where:
- `1` → High risk of heart disease
- `0` → Low risk of heart disease

---

## Dataset
- Publicly available heart disease dataset (UCI-style structured clinical dataset)
- Features include demographic, physiological, and test-based attributes such as:
  - Age, sex
  - Chest pain type
  - Resting blood pressure
  - Serum cholesterol
  - Maximum heart rate
  - Exercise-induced angina
  - ECG results, ST depression, etc.

Target variable indicates presence or absence of heart disease.

---

## Machine Learning Models Trained
Multiple classical ML models were trained and evaluated to compare performance and trade-offs between accuracy and interpretability:

- Logistic Regression
- Ridge Classifier
- Naive Bayes
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)
- Decision Tree
- Random Forest
- Gradient Boosting
- LightGBM (LGBM)
- XGBoost

All trained models are saved in the `Sav_models/` directory as serialized `.sav` files.

---

## Model Selection Rationale
While multiple models were evaluated, **Logistic Regression** is treated as the primary reference model due to:
- High interpretability (feature coefficients map directly to risk contribution)
- Well-calibrated probability outputs
- Lower overfitting risk on small/medium clinical datasets

Tree-based ensemble models (Random Forest, Gradient Boosting, LGBM, XGBoost) were used for performance comparison but not prioritized over interpretability.

---

## Evaluation Approach
Models were evaluated using standard classification metrics such as:
- Accuracy
- Precision & Recall
- ROC-AUC

Special emphasis was placed on recall, as **false negatives in healthcare applications carry higher risk**.

---

## Project Structure
```
Heart-Disease-Risk-Prediction/
├── Sav_models/              # Trained model artifacts (.sav)
├── notebooks/               # Data exploration & model training notebooks
├── streamlit_app.py         # Local Streamlit application
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Running the Application Locally
This project includes a **local Streamlit interface** for testing and demonstration purposes. The app is **not deployed publicly**.

### Steps
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Key Takeaways
- Demonstrates a complete ML pipeline from data to prediction
- Focuses on **healthcare-appropriate trade-offs** (interpretability vs performance)
- Emphasizes reproducibility and honest model reporting
- Designed as an internship-level, interview-defensible ML project

---

## Author
**Manas Ranjan**

This project was adapted, cleaned, and extended for educational and portfolio purposes.
