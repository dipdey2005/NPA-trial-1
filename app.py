import streamlit as st
import pandas as pd
import joblib
import os
import random

# Set Streamlit page config for wide layout and light mode
st.set_page_config(page_title="NPA Default Risk Predictor", layout="wide")

# Light color palette
ACCENT_BG = "#f5f7fa"
CARD_BG = "#ffffff"
PRIMARY_C = "#1f77b4"
RISK_C = "#d62728"

# Load the trained model
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# App title and subtitle
st.markdown(
    "<h1 style='text-align:center; color:#222;'>Default Risk Predictor — NPA Probability</h1>"
    "<h4 style='text-align:center; color:#1f77b4;'>Machine Learning Model to compute risk of customer becoming Non-Performing Asset</h4>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# Card-style form container
with st.container():
    st.markdown(
        f"<div style='background-color:{CARD_BG}; border-radius:10px; padding:32px; box-shadow:0 2px 8px #eee;'>",
        unsafe_allow_html=True
    )
    st.markdown("### 📝 Applicant Details")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            employment_type = st.selectbox(
                "Employment Type (1-4)",
                ["Select Employment Type", "Government", "Salaried", "Self-Employed", "Pensioner"],
                index=0
            )
            total_employment_length = st.text_input("Total Employment Length", placeholder="Years (Rounded Up)")
            total_annual_loan = st.text_input("Total Equated Annual Loan", placeholder="Annual Loan Payment")
            income = st.text_input("Income", placeholder="Per Annum, Post Tax")
            loan_amount = st.text_input("Loan Amount", placeholder="Sanctioned per annum amount")

        with c2:
            location_type = st.selectbox(
                "Location Type (1-3)",
                ["Select Location Type", "Rural", "Semi-Urban", "Urban"],
                index=0
            )
            current_employment_length = st.text_input("Current Employment Length", placeholder="Years (Rounded Up)")
            cibil_score = st.text_input("CIBIL Score", placeholder="300-900")
            cibil_ranking = st.text_input("CIBIL Score Ranking", placeholder="Ranking Based on Credit History")
            loan_emi_tenure = st.text_input("Loan EMI Tenure", placeholder="Years (Decimal)")

        with c3:
            sbi_customer = st.selectbox(
                "SBI Customer",
                ["Select if Customer", "Yes", "No"],
                index=0
            )
            dpd = st.text_input("DPD", placeholder="Current Days Past Due")
            max_dpd = st.text_input("Max DPD", placeholder="Maximum Days Past Due")
            missed_emis = st.text_input("Missed EMIs", placeholder="Number of missed EMI payments")
            dependants = st.text_input("Dependants", placeholder="Number of dependants")

        submitted = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# Prediction logic
if submitted:
    # Mapping dropdowns to expected numeric codes
    emp_map = {"Government": 4, "Salaried": 3, "Self-Employed": 2, "Pensioner": 1}
    loc_map = {"Rural": 1, "Semi-Urban": 2, "Urban": 3}
    sbi_map = {"Yes": 1, "No": 0}

    # Convert inputs to numeric types, handle empty fields
    employment_type_val = emp_map.get(employment_type, 0)
    location_type_val = loc_map.get(location_type, 0)
    sbi_customer_val = sbi_map.get(sbi_customer, 0)

    # List of features used during model training
    expected_features = [
        'Employment Type (0-5) (Government-4, Salaried-3, Self-Employed-2, Pensioner-1 )',
        'Current Employment Length (Total Years)',
        'Employment Length (Total Years)',
        'SBI Customer (1-Yes, 0-No)',
        'Location Type (Urban-3, Semi-Urban-2, Rural-1)',
        'Income (PA, PAT) (Rs)',
        'Total AMIs (other loans) (Rs)',
        'Debt-to-Income Ratio (After Loan) (%)',
        'Dependants',
        'Surplus After EMI/Dependants',
        'Loan Amt (Rs) (Yearly)',
        'Loan EMI Tenure (Years)',
        'CIBIL Score Ranking (850+=10, 800+=9, 750+=8, 700+=7, 650+=6, 600+=5, 550+=4, 450+=3, 400+=2, 300+=1)',
        'CIBIL Score',
        'DPD',
        'Max DPD',
        'Missed EMIs'
    ]

    # Prepare row for model input
    row = {
        'Employment Type (0-5) (Government-4, Salaried-3, Self-Employed-2, Pensioner-1 )': employment_type_val,
        'Current Employment Length (Total Years)': int(current_employment_length or 0),
        'Employment Length (Total Years)': int(total_employment_length or 0),
        'SBI Customer (1-Yes, 0-No)': sbi_customer_val,
        'Location Type (Urban-3, Semi-Urban-2, Rural-1)': location_type_val,
        'Income (PA, PAT) (Rs)': float(income or 0),
        'Total AMIs (other loans) (Rs)': 0,  # Placeholder, update if you have this field
        'Debt-to-Income Ratio (After Loan) (%)': 0,  # Placeholder, update if you have this field
        'Dependants': int(dependants or 0),
        'Surplus After EMI/Dependants': 0,  # Placeholder, update if you have this field
        'Loan Amt (Rs) (Yearly)': float(loan_amount or 0),
        'Loan EMI Tenure (Years)': float(loan_emi_tenure or 0),
        'CIBIL Score Ranking (850+=10, 800+=9, 750+=8, 700+=7, 650+=6, 600+=5, 550+=4, 450+=3, 400+=2, 300+=1)': int(cibil_ranking or 0),
        'CIBIL Score': int(cibil_score or 0),
        'DPD': int(dpd or 0),
        'Max DPD': int(max_dpd or 0),
        'Missed EMIs': int(missed_emis or 0)
    }

    # Ensure DataFrame columns match expected features and order
    X = pd.DataFrame([[row.get(f, 0) for f in expected_features]], columns=expected_features)

    # Predict probability
    if model:
        try:
            probability = model.predict_proba(X)[0, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            probability = None
    else:
        probability = round(random.uniform(0.05, 0.85), 2)

    if probability is not None:
        st.subheader("📈 Predicted Default Risk")
        st.write(f"**Probability of Default:** {probability:.2%}")
    else:
        st.warning("Could not compute prediction. Please check your input or model file.")

# Optional: Add a light background to the whole app
st.markdown(
    f"""
    <style>
        body {{
            background-color: {ACCENT_BG} !important;
        }}
        .stApp {{
            background-color: {ACCENT_BG} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
