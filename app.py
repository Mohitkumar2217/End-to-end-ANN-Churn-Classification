import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load the trained model 
model = tf.keras.models.load_model('model.h5')

## Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit App UI
st.title('Customer Churn Prediction')

# User Input Controls
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 35)
balance = st.number_input('Balance', min_value=0.0, value=50000.0, step=1000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
tenure = st.slider('Tenure (Years)', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare base numeric and label-encoded features
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

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine numeric/label features with one-hot encoded columns
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Align column order with scaler expectations
if hasattr(scaler, 'feature_names_in_'):
    input_df = input_df[scaler.feature_names_in_]

# Scale the features
input_scaled = scaler.transform(input_df)

# Predict churn
prediction = model.predict(input_scaled)
prediction_proba = float(prediction[0][0])

st.divider()
st.subheader("Prediction Result")
st.write(f"**Churn Probability:** `{prediction_proba:.2%}`")

if prediction_proba > 0.5:
    st.error('⚠️ **The customer is likely to churn.**')
else:
    st.success('✅ **The customer is not likely to churn.**')