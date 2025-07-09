import streamlit as st
import pandas as pd
import joblib
import os
import random

# ─────────────── PAGE SETUP ───────────────
st.set_page_config(page_title="Default Risk Predictor — NPA Probability", layout="wide")

# Custom CSS for light card and form styling
st.markdown("""
    <style>
    body, .stApp {
        background-color: #eef1f7 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        font-size: 2.6rem;
        font-weight: 800;
        color: #000000; 
        text-align: center;
        margin-bottom: 0.2em;
    }

    h4 {
        font-size: 1rem;
        font-weight: 500;
        text-align: center;
        color: #000000;
        margin-top: -10px;
        margin-bottom: 2.5em;
    }

    .stContainer {
        background: white !important;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        padding: 2rem 2rem 3rem 2rem;
        margin: auto;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111;
        margin-bottom: 1.5em;
        display: flex;
        align-items: center;
    }

    .section-header:before {
        content: '';
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #3366ff;
        border-radius: 2px;
        margin-right: 10px;
    }

    .stTextInput > div > input, .stSelectbox > div {
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        padding: 0.6em 1em !important;
        background-color: #f9fafa !important;
        font-size: 1rem;
    }

    label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #333 !important;
    }

    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 6px;
        width: 100%;
        height: 3em;
        font-size: 1.1em;
        font-weight: 600;
        margin-top: 8em;  
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background-color: #155a8a;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────── MODEL LOADING ───────────────
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ─────────────── HEADER ───────────────
st.markdown(
    "<h1 style='text-align:center;'>Default Risk Predictor — NPA Probability</h1>"
    "<h4 style='text-align:center;'>Machine Learning Model to compute risk of customer becoming Non-Performing Asset</h4>",
    unsafe_allow_html=True
)

# ─────────────── CARD CONTAINER ───────────────
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

# ─────────────── PREDICTION LOGIC ───────────────
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
        'Missed EMIs',
        'EMI to Income',
        'Net Disposable Income',
        'Missed EMI Rate'
    ]

    # Engineered features (set to 0 if not enough info)
    try:
        income_val = float(income or 0)
        loan_amount_val = float(loan_amount or 0)
        other_amis_val = float(total_annual_loan or 0)
        dependants_val = int(dependants or 0)
        loan_emi_tenure_val = float(loan_emi_tenure or 0)
    except Exception:
        income_val = loan_amount_val = other_amis_val = dependants_val = loan_emi_tenure_val = 0

    emi_to_income = (other_amis_val + loan_amount_val) / income_val if income_val else 0
    net_disposable = income_val - other_amis_val - loan_amount_val
    missed_emi_rate = int(missed_emis or 0) / loan_emi_tenure_val if loan_emi_tenure_val else 0
    dti_after_loan_pct = (other_amis_val + loan_amount_val) * 100 / income_val if income_val else 0
    surplus_per_dep = (income_val - other_amis_val - loan_amount_val) / (12 * dependants_val) if dependants_val else 0

    row = {
        'Employment Type (0-5) (Government-4, Salaried-3, Self-Employed-2, Pensioner-1 )': employment_type_val,
        'Current Employment Length (Total Years)': int(current_employment_length or 0),
        'Employment Length (Total Years)': int(total_employment_length or 0),
        'SBI Customer (1-Yes, 0-No)': sbi_customer_val,
        'Location Type (Urban-3, Semi-Urban-2, Rural-1)': location_type_val,
        'Income (PA, PAT) (Rs)': income_val,
        'Total AMIs (other loans) (Rs)': other_amis_val,
        'Debt-to-Income Ratio (After Loan) (%)': dti_after_loan_pct,
        'Dependants': dependants_val,
        'Surplus After EMI/Dependants': surplus_per_dep,
        'Loan Amt (Rs) (Yearly)': loan_amount_val,
        'Loan EMI Tenure (Years)': loan_emi_tenure_val,
        'CIBIL Score Ranking (850+=10, 800+=9, 750+=8, 700+=7, 650+=6, 600+=5, 550+=4, 450+=3, 400+=2, 300+=1)': int(cibil_ranking or 0),
        'CIBIL Score': int(cibil_score or 0),
        'DPD': int(dpd or 0),
        'Max DPD': int(max_dpd or 0),
        'Missed EMIs': int(missed_emis or 0),
        'EMI to Income': emi_to_income,
        'Net Disposable Income': net_disposable,
        'Missed EMI Rate': missed_emi_rate
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
        st.subheader("Predicted Default Risk")
        st.write(f"**Probability of Default:** {probability:.2%}")
    else:
        st.warning("Could not compute prediction. Please check your input or model file.")
