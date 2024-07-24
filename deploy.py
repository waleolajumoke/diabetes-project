import streamlit as st
import pandas as pd
import numpy as np

# Load model
import joblib
model = joblib.load('model.pkl')


# Load scaler
sc = joblib.load('scaler.pkl')

# create a function to predict diabetes 
def diabetes_prediction(input_data):
    features = sc.transform(input_data)
    pred = model.predict(features)
    return int(pred[0])


# designing the form
st.title('Diabetes Prediction')
pregnacy = st.number_input('Number of times pregnant', 0, 20)
glucose = st.number_input('Enter Plasma glucose concentration', 0, 200, 1)
blood_pressure = st.number_input('Enter Diastolic blood pressure', 0, 130, 1)
skin = st.number_input('Enter Triceps skin fold thickness', 0, 100, 1)
insulin = st.number_input('Enter 2-Hour serum insulin', 0, 200 )
bmi = st.number_input('Enter Body mass index', 0.0, 70.0, 1.0)
dpf = st.number_input('Enter Diabetes pedigree function', 0.0, 2.5)
age = st.number_input('Enter Age', 0, 120, 1)

# submit form
if st.button('Predict'):
    output = diabetes_prediction([[pregnacy, glucose, blood_pressure, skin, insulin, bmi, dpf, age]])
    if output == 0:
        st.success('The person is not diabetic')
    else:
        st.success('The person is diabetic')