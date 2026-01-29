import pickle
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu


@st.cache_resource
def load_models():
    return {
        "Logistic Regression": pickle.load(open("Sav_models/logistic_regression.sav", "rb")),
        "Ridge Classifier": pickle.load(open("Sav_models/ridge.sav", "rb")),
        "Naive Bayes": pickle.load(open("Sav_models/naive_bayes.sav", "rb")),
        "Support Vector Machine (SVM)": pickle.load(open("Sav_models/svm.sav", "rb")),
        "K-Nearest Neighbors (KNN)": pickle.load(open("Sav_models/knn.sav", "rb")),
        "Linear Discriminant Analysis (LDA)": pickle.load(open("Sav_models/linear_discriminant.sav", "rb")),
        "Decision Tree": pickle.load(open("Sav_models/decision_tree.sav", "rb")),
        "Random Forest": pickle.load(open("Sav_models/random_forest.sav", "rb")),
        "Gradient Boosting": pickle.load(open("Sav_models/gradient_boosting.sav", "rb")),
        "LightGBM (LGBM)": pickle.load(open("Sav_models/lgbm.sav", "rb")),
        "XGBoost": pickle.load(open("Sav_models/xgboost.sav", "rb")),
    }


def predict_risk(model, features):
    return model.predict(features)[0]


def show_user_inputs():
    st.title("Heart Disease Risk Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1)
        trestbps = st.number_input("Resting Blood Pressure", min_value=1)
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        oldpeak = st.number_input("ST Depression")
        thal = st.selectbox("Thal", [3, 6, 7])

    with col2:
        sex = st.selectbox("Sex", [0, 1])
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=1)
        thalach = st.number_input("Max Heart Rate", min_value=1)
        slope = st.selectbox("Slope", [1, 2, 3])

    with col3:
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])

    return [[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]]


def main():
    st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

    models = load_models()
    model_names = ["All Models"] + list(models.keys())

    with st.sidebar:
        selected_model = option_menu(
            "Select Model",
            model_names,
            icons=["heart"] * len(model_names),
            default_index=0
        )

    features = show_user_inputs()

    if st.button("Predict"):
        if selected_model == "All Models":
            results = {
                name: predict_risk(model, features)
                for name, model in models.items()
            }

            df = pd.DataFrame(results.items(), columns=["Model", "Prediction"])
            df["Prediction"] = df["Prediction"].astype(str)

            fig = px.bar(
                df,
                x="Model",
                y="Prediction",
                color="Prediction",
                color_discrete_map={"1": "red", "0": "green"},
                title="Model-wise Heart Disease Prediction"
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        else:
            result = predict_risk(models[selected_model], features)
            st.subheader(selected_model)
            if result == 1:
                st.error("⚠️ High risk of heart disease")
            else:
                st.success("✅ Low risk of heart disease")


if __name__ == "__main__":
    main()
