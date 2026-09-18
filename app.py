import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# ----------------------------
# Load Model and Encoders
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")
    onehot_encoder_geo = joblib.load("onehot_encoder_geo.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, onehot_encoder_geo, scaler

model, onehot_encoder_geo, scaler = load_artifacts()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a bank customer is likely to churn based on their profile.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", list(onehot_encoder_geo.categories_[0]) + ["Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure (years with bank)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, value=1000.0, step=100.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Churn"):
    try:
        # ----- Handle Geography One-Hot -----
        geo_categories = onehot_encoder_geo.categories_[0]
        if geography in geo_categories:
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        else:
            geo_encoded = np.zeros((1, len(geo_categories)))

        # Safe DataFrame creation
        try:
            feature_names = onehot_encoder_geo.get_feature_names_out(["Geography"])
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=feature_names)
        except ValueError:
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=[f"geo_{i}" for i in range(geo_encoded.shape[1])])

        # ----- Encode Gender -----
        gender_val = 1 if gender == "Male" else 0

        # ----- Numeric features -----
        numeric_df = pd.DataFrame([[
            credit_score, gender_val, age, tenure, balance,
            num_of_products, has_cr_card, is_active_member, estimated_salary
        ]], columns=[
            "CreditScore", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ])

        # ----- Combine everything -----
        final_input = pd.concat([numeric_df, geo_encoded_df], axis=1)

        # ----- Scale -----
        final_scaled = scaler.transform(final_input)

               # ----- DEBUG -----
        st.write("DEBUG INPUT:", final_input)
        st.write("DEBUG SCALED INPUT:", final_scaled)

        # ----- Predict -----
        prediction_prob = model.predict(final_scaled)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0

        # ----- Display -----
        if prediction == 1:
            st.error(f"⚠️ This customer is likely to churn (Probability: {prediction_prob:.2%})")
        else:
            st.success(f"✅ This customer is unlikely to churn (Probability: {1 - prediction_prob:.2%})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
