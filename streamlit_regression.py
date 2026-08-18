import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Page Configuration
st.set_page_config(
    page_title="Customer Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Cache model and encoders in memory
@st.cache_resource
def load_artifacts():
    loaded_model = tf.keras.models.load_model('salary_model.h5')
    
    with open('salary_onehot_encoder_geo.pkl', 'rb') as f:
        geo_encoder = pickle.load(f)
    with open('salary_label_encoder_gender.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)
    with open('salary_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
        
    return loaded_model, geo_encoder, gender_encoder, loaded_scaler

model, onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()

# App Header
st.title("💼 Estimated Salary Prediction")
st.markdown("Predict a customer's expected annual salary using demographic and account risk indicators.")

# User Inputs Form
with st.form("salary_prediction_form"):
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("Gender", label_encoder_gender.classes_)
        age = st.slider("Age", 18, 92, 35)
        tenure = st.slider("Tenure (Years)", 0, 10, 3)
        num_of_products = st.slider("Number of Products", 1, 4, 1)

    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
        has_cr_card = st.selectbox("Has Credit Card", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox("Is Active Member", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        exited = st.selectbox("Customer Churned / High Risk (Exited)", options=[0, 1], format_func=lambda x: "Yes (Exited)" if x == 1 else "No (Retained)")

    submit_button = st.form_submit_button("Predict Estimated Salary", use_container_width=True)

# Prediction Logic
if submit_button:
    with st.spinner("Calculating estimated salary..."):
        # 1. Base input dataframe (excluding one-hot Geography)
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'Exited': [exited]
        })

        # 2. One-hot encode Geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # 3. Concatenate all features
        input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # 4. Enforce exact column ordering from scaler training
        if hasattr(scaler, 'feature_names_in_'):
            input_df = input_df[scaler.feature_names_in_]

        # 5. Scale features
        input_scaled = scaler.transform(input_df)

        # 6. Predict Salary
        predicted_salary = float(model.predict(input_scaled, verbose=0)[0][0])
        predicted_salary = max(0.0, predicted_salary)  # Ensure non-negative value

    # Display Result
    st.divider()
    st.subheader("Prediction Result")
    st.metric(
        label="Estimated Annual Salary",
        value=f"${predicted_salary:,.2f}"
    )