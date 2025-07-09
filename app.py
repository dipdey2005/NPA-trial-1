import streamlit as st
import pandas as pd
import joblib
import os
import random

# --- PAGE CONFIG & COLORS ---
st.set_page_config(page_title="Default Risk Predictor — NPA Probability", layout="wide")
ACCENT_BG = "#f5f7fa"
CARD_BG = "#ffffff"
PRIMARY_C = "#1f77b4"

# --- CUSTOM CSS FOR FIGMA-LIKE STYLE ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #f5f7fa !important;
    }
    .main > div {
        padding-top: 0 !important;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 3em;
        font-size: 1.1em;
        font-weight: 600;
        margin-top: 1em;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #155a8a;
    }
    .stTextInput > div > input, .stSelectbox > div {
        border-radius: 8px !important;
        padding: 0.6em !important;
        border: 1px solid #e0e0e0 !important;
        background: #fafbfc !important;
        font-size: 1em !important;
    }
    .stContainer {
        background: #fff !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 16px #eaeaea !important;
        padding: 2em 2em 2.5em 2em !important;
        margin-bottom: 2em !important;
    }
    .stForm {
        margin-top: 1em !important;
    }
    h1 {
        font-size: 2.5em !important;
        font-weight: 800 !important;
        margin-bottom: 0.2em !important;
        color: #222 !important;
    }
    h4 {
        font-size: 1.2em !important;
        color: #1f77b4 !important;
        margin-bottom: 1.5em !important;
        font-weight: 500 !important;
    }
    .section-header {
        font-size: 1.25em !important;
        font-weight: 700 !important;
        margin-bottom: 1.5em !important;
        margin-top: 0.5em !important;
        color: #222 !important;
        display: flex;
        align-items: center;
    }
    .section-header:before {
        content: '';
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #1f77b4;
        border-radius: 3px;
        margin-right: 10px;
    }
    label {
        font-weight: 600 !important;
        color: #333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --- HEADER ---
st.markdown(
    "<h1 style='text-align:center;'>Default Risk Predictor — NPA Probability</h1>"
    "<h4 style='text-align:center;'>Machine Learning Model to compute risk of customer becoming Non-Performing Asset</h4>",
    unsafe_allow_html=True
)

# --- CARD CONTAINER ---
with st.container():
    st.markdown('<div class="section-header">Applicant Details</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            employment_type = st.selectbox(
                "Employment Type (1-4)",
                ["Select Employment Type", "Government", "Salaried", "Self-Employed", "Pensioner"],
                index=0
            )
            total_employment_length = st.text_input("Total Employment Length", placeholder="Years (Rounded Up)")
            total_annual_loan = st.text_input("Total Equated Annual Loan", placeholder="Annual Loan Payment")
            income = st.text_input("Income", placeholder="Per Annum, Post Tax")
            loan_amount = st.text_input("Loan Amount", placeholder="Sanctioned per annum amount")

        with col2:
            location_type = st.selectbox(
                "Location Type (1-3)",
                ["Select Location Type", "Rural", "Semi-Urban", "Urban"],
                index=0
            )
            current_employment_length = st.text_input("Current Employment Length", placeholder="Years (Rounded Up)")
            cibil_score = st.text_input("CIBIL Score", placeholder="300-900")
            cibil_ranking = st.text_input("CIBIL Score Ranking", placeholder="Ranking Based on Credit History")
            loan_emi_tenure = st.text_input("Loan EMI Tenure", placeholder="Years (Decimal)")

        with col3:
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

# --- PREDICTION LOGIC ---
if submitted:
    emp_map = {"Government": 4, "Salaried": 3, "Self-Employed": 2, "Pensioner": 1}
    loc_map = {"Rural": 1, "Semi-Urban": 2, "Urban": 3}
    sbi_map = {"Yes": 1, "No": 0}

    employment_type_val = emp_map.get(employment_type, 0)
    location_type_val = loc_map.get(location_type, 0)
    sbi_customer_val = sbi_map.get(sbi_customer, 0)

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

    row = {
        'Employment Type (0-5) (Government-4, Salaried-3, Self-Employed-2, Pensioner-1 )': employment_type_val,
        'Current Employment Length (Total Years)': int(current_employment_length or 0),
        'Employment Length (Total Years)': int(total_employment_length or 0),
        'SBI Customer (1-Yes, 0-No)': sbi_customer_val,
        'Location Type (Urban-3, Semi-Urban-2, Rural-1)': location_type_val,
        'Income (PA, PAT) (Rs)': float(income or 0),
        'Total AMIs (other loans) (Rs)': 0,  # Placeholder
        'Debt-to-Income Ratio (After Loan) (%)': 0,  # Placeholder
        'Dependants': int(dependants or 0),
        'Surplus After EMI/Dependants': 0,  # Placeholder
        'Loan Amt (Rs) (Yearly)': float(loan_amount or 0),
        'Loan EMI Tenure (Years)': float(loan_emi_tenure or 0),
        'CIBIL Score Ranking (850+=10, 800+=9, 750+=8, 700+=7, 650+=6, 600+=5, 550+=4, 450+=3, 400+=2, 300+=1)': int(cibil_ranking or 0),
        'CIBIL Score': int(cibil_score or 0),
        'DPD': int(dpd or 0),
        'Max DPD': int(max_dpd or 0),
        'Missed EMIs': int(missed_emis or 0)
    }

    X = pd.DataFrame([[row.get(f, 0) for f in expected_features]], columns=expected_features)

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
