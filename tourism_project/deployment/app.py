import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_pred.joblib")
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App - Visit with Us ")
#st.write("The Customer Churn Prediction App is an internal tool for bank staff that predicts whether customers are at risk of churning based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to Purchase the Wellness Tourism Package.")

# Collect user input
Age = st.number_input("Age(customer's Age)", min_value=0, max_value=100, value=20)
TypeofContact = st.selectbox("TypeofContact (Company Invited or Self Inquiry)", ["Company Invited","Self Inquiry"])
CityTier = st.selectbox("CityTier (customer's City i.e, Tier1/2/3)", ["1","2","3")
Occupation = st.selectbox("Occupation (Customer's occupation)", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
Gender = st.selectbox("Gender (customer's gender)",['Female', 'Male', 'Fe Male'])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting (number of person accompanied on the trip)", min_value=1, max_value=5, value=1)
PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer:", ["3","4","5"])
MaritalStatus = st.selectbox("Marital status?", ['Single', 'Divorced', 'Married', 'Unmarried'])
NumberOfTrips = st.number_input("NumberOfTrips(Number of trips a customer takes annually)", min_value=1,max_value=22, value=5)
Passport = st.selectbox("Passport(Does customer have passport 0:No, 1:Yes?)", ["Yes","No"])
OwnCar = st.number_input("OwnCar(Does customer have owncar  (0: No, 1: Yes)?)", ["Yes","No"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting(Number of children below 5 yrs visiting)", min_value=1,max_value=3, value=1)
Designation = st.selectbox("Designation(customer's Designation)", [['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP']])
MonthlyIncome = st.number_input("MonthlyIncome(customer's Monthly Income)", min_value=1000,max_value=100000,value=1000)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport=="Yes" else 0,
    "OwnCar": 1 if OwnCar=="Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
   }])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
