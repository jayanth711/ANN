import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler 
with open('geo_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

geography_categories = ['France', 'Germany', 'Spain']

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', geography_categories)  # Correct usage of the encoder
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]  # Add Geography to the DataFrame
})

#  One-hot encode 'Geography' manually based on known categories
geo_encoded = np.zeros(len(geography_categories))
geo_encoded[geography_categories.index(geography)] = 1
geo_encoded_df = pd.DataFrame([geo_encoded], columns=[f'Geography_{cat}' for cat in geography_categories])

# Combine one-hot encoded columns with input data
input_df = pd.concat([input_data.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

# Display result
if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

## display the probability
st.write(f'Churn Probability: {prediction_prob:.2f}')