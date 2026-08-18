import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Cache model and artifacts to avoid reloading from disk on every user interaction
@st.cache_resource
def load_artifacts():
    # Load ANN model
    loaded_model = tf.keras.models.load_model('model.h5')
    
    # Load encoders and scaler
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_geo = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
        
    return loaded_model, onehot_geo, le_gender, loaded_scaler

model, onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()

# App Header
st.title('🏦 Customer Churn Prediction')
st.markdown("Enter customer details below to estimate churn risk.")

# User Inputs Form / Layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 35)
    tenure = st.slider('Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)

with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650, step=1)
    balance = st.number_input('Balance ($)', min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0)
    has_cr_card = st.selectbox('Has Credit Card', options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox('Is Active Member', options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction trigger button
if st.button('Predict Churn Risk', type='primary', use_container_width=True):
    with st.spinner('Calculating probability...'):
        # Base DataFrame
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode Geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # Combine tabular features
        input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Enforce training column order
        if hasattr(scaler, 'feature_names_in_'):
            input_df = input_df[scaler.feature_names_in_]

        # Scale features and run inference
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled, verbose=0)
        prediction_proba = float(prediction[0][0])

    # Display Results
    st.divider()
    st.subheader("Prediction Result")
    
    col_metric, col_status = st.columns([1, 2])
    with col_metric:
        st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
    with col_status:
        if prediction_proba > 0.5:
            st.error('⚠️ **High Risk:** The customer is likely to churn.')
        else:
            st.success('✅ **Low Risk:** The customer is not likely to churn.')