import streamlit as st 
import pandas as pd
import joblib
import os
import random

# ─────────────── PAGE SETUP ───────────────
st.set_page_config(page_title="Default Risk Predictor — NPA Probability", layout="wide")

# Custom Light Theme CSS
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
    }
    ul {
        list-style-type: none;
        padding-left: 0;
        font-size: 16px;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────── LOAD MODEL ───────────────
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
st.sidebar.info("✅ Model loaded" if model else "⚠️ Model not found — using mock outputs")

# ─────────────── HEADER ───────────────
st.markdown("## 🏦 Bank Default Risk Predictor")
st.markdown("A smarter way to assess applicant credit risk using ML & financial signals.")
st.markdown("---")

# ─────────────── FORM ───────────────
with st.form("prediction_form"):
    st.markdown("### 📋 Applicant Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        employment_type          = st.selectbox("💼 Employment Type (0–5)", list(range(6)))
        current_employment_years = st.slider("🧑‍💼 Current Employment (yrs)", 0, 40, 5)
        total_employment_years   = st.slider("📈 Total Employment (yrs)", 0, 40, 10)
        sbi_customer             = st.selectbox("🏛 SBI Customer", [0, 1])
        location_type            = st.selectbox("📍 Location Type (1–Rural → 3–Urban)", [1, 2, 3])
        dependants               = st.slider("👨‍👩‍👦 Dependants", 1, 10, 2)

    with c2:
        income        = st.slider("💰 Annual Income (₹)", 1_00_000, 50_00_000, 5_00_000, 50_000)
        other_amis    = st.slider("💸 Other Loan AMIs (₹)", 0, 20_00_000, 1_00_000, 10_000)
        loan_amount   = st.slider("🏦 Loan Amount / yr (₹)", 10_000, 20_00_000, 2_00_000, 10_000)
        loan_tenure   = st.slider("📅 Loan Tenure (yrs)", 1, 30, 5)
        cibil_rank    = st.slider("📊 CIBIL Rank (1–10)", 1, 10, 7)

    with c3:
        cibil_score = st.slider("📉 CIBIL Score", 300, 900, 750, 10)
        dpd         = st.slider("⏱ Days Past Due (DPD)", 0, 1000, 0, 10)
        max_dpd     = st.slider("🚨 Max DPD", 0, 1000, 0, 10)
        missed_emis = st.slider("❌ Missed EMIs", 0, 60, 0)

    submitted = st.form_submit_button("🔍 Predict")

# ─────────────── PREDICTION + OUTPUT ───────────────
if submitted:
    emi_to_income      = (other_amis + loan_amount) / income if income else 0
    net_disposable     = income - other_amis - loan_amount
    missed_emi_rate    = missed_emis / loan_tenure if loan_tenure else 0
    dti_after_loan_pct = (other_amis + loan_amount) * 100 / income if income else 0
    surplus_per_dep    = (income - other_amis - loan_amount) / (12 * dependants) if dependants else 0

    row = {
        'Employment Type (0-5) (Government-4, Salaried-3, Self-Employed-2, Pensioner-1 )': employment_type,
        'Current Employment Length (Total Years)': current_employment_years,
        'Employment Length (Total Years)': total_employment_years,
        'SBI Customer (1-Yes, 0-No)': sbi_customer,
        'Location Type (Urban-3, Semi-Urban-2, Rural-1)': location_type,
        'Income (PA, PAT) (Rs)': income,
        'Total AMIs (other loans) (Rs)': other_amis,
        'Debt-to-Income Ratio (After Loan) (%)': dti_after_loan_pct,
        'Dependants': dependants,
        'Surplus After EMI/Dependants': surplus_per_dep,
        'Loan Amt (Rs) (Yearly)': loan_amount,
        'Loan EMI Tenure (Years)': loan_tenure,
        'CIBIL Score Ranking (850+=10...)': cibil_rank,
        'CIBIL Score': cibil_score,
        'DPD': dpd,
        'Max DPD': max_dpd,
        'Missed EMIs': missed_emis,
        'EMI to Income': emi_to_income,
        'Net Disposable Income': net_disposable,
        'Missed EMI Rate': missed_emi_rate
    }

    X = pd.DataFrame([row])
    probability = model.predict_proba(X)[0,1] if model else round(random.uniform(0.05, 0.85), 2)
    pct = f"{probability:.0%}"
    risk_color = "#d62728" if probability > 0.5 else "#1f77b4"

    st.markdown("### 📈 Predicted Default Risk")
    st.markdown(f"""
    <div style='
        background: linear-gradient(to right, {risk_color} {probability*100:.0f}%, #dee2e6 {probability*100:.0f}%);
        height: 30px;
        border-radius: 8px;
        margin-bottom: 10px;
        position: relative;
    '>
        <div style='
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 18px;
            color: white;
        '>{pct}</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────── SUMMARY ───────────────
    st.markdown("---")
    st.subheader("🔍 Summary Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Key Inputs")
        st.markdown(f"""
        <ul>
            <li>💰 <b>Income:</b> ₹{income:,}</li>
            <li>🏦 <b>Loan Amount:</b> ₹{loan_amount:,}</li>
            <li>📉 <b>Other AMIs:</b> ₹{other_amis:,}</li>
            <li>🔢 <b>CIBIL Score:</b> {cibil_score}</li>
            <li>⏱️ <b>DPD (Days):</b> {dpd}</li>
            <li>❌ <b>Missed EMIs:</b> {missed_emis}</li>
        </ul>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧮 Engineered Features")
        st.markdown(f"""
        <ul>
            <li>📊 <b>EMI to Income Ratio:</b> {emi_to_income:.2f}</li>
            <li>💸 <b>Net Disposable Income:</b> ₹{net_disposable:,.2f}</li>
            <li>📅 <b>Missed EMI Rate:</b> {missed_emi_rate:.2f}</li>
            <li>📉 <b>Debt-to-Income (%):</b> {dti_after_loan_pct:.2f}%</li>
            <li>👨‍👩‍👧 <b>Surplus / Dependant / Month:</b> ₹{surplus_per_dep:,.2f}</li>
        </ul>
        """, unsafe_allow_html=True)
