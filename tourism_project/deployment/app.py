"""
Streamlit app for the Wellness Tourism Package prediction model.
Loads the model committed to this folder by the training pipeline, collects
customer inputs, and displays whether the customer is likely to purchase
the package.
"""

import pandas as pd
import streamlit as st
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧭")

st.title("🧭 Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer's details below to predict whether they are likely "
    "to purchase the new Wellness Tourism Package."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    type_of_contact = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, value=20000.0, step=500.0
    )

with col2:
    num_person_visiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2
    )
    num_children_visiting = st.number_input(
        "Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0
    )
    num_trips = st.number_input(
        "Average Number of Trips per Year", min_value=0.0, value=3.0, step=1.0
    )
    passport = st.selectbox("Holds Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    num_followups = st.number_input(
        "Number of Follow-ups", min_value=0.0, value=3.0, step=1.0
    )
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=0.0, value=10.0, step=1.0
    )
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.success(f"✅ Likely to purchase the package (probability: {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase the package (probability: {probability:.1%})")
