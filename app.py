import streamlit as st 
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os, random

# ─────────────────────────────  page setup  ──────────────────────────────
st.set_page_config(layout="wide")
ACCENT_BG = "#ffffff"        # light background
PRIMARY_C = "#1f77b4"        # blue
RISK_C    = "#d62728"        # red
BAR_C     = "#00d491"        # green

# CSS for light mode
st.markdown(
    f"""
    <style>
    body, .reportview-container, .sidebar .sidebar-content {{
        background-color: {ACCENT_BG};
        color: black;
    }}
    h1, h2, h3, h4, h5, h6, label, p, .stSlider, .stSelectbox, .stNumberInput {{
        color: black;
    }}
    </style>""",
    unsafe_allow_html=True
)

# ─────────────────────────────  load / fallback model  ───────────────────
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
st.sidebar.info("✅ Model loaded" if model else "⚠️  Model not found – mock outputs")

# ─────────────────────────────  title  ───────────────────────────────────
st.title("Bank Default Risk Predictor — NPA Probability")

# ─────────────────────────────  form  ────────────────────────────────────
with st.form("prediction_form"):
    st.header("📋 Applicant Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        employment_type          = st.selectbox("Employment Type (0–5)", list(range(6)))
        current_employment_years = st.slider("Current Employment Length (yrs)", 0, 40, 5)
        total_employment_years   = st.slider("Total Employment Length (yrs)",   0, 40, 10)
        sbi_customer             = st.selectbox("SBI Customer", [0, 1])
        location_type            = st.selectbox("Location Type (1 Rural – 3 Urban)", [1, 2, 3])
        dependants               = st.slider("Dependants", 1, 10, 2)

    with c2:
        income        = st.slider("Annual Income (₹)", 1_00_000, 50_00_000, 5_00_000, 50_000)
        other_amis    = st.slider("Other‑loan AMIs (₹)", 0, 20_00_000, 1_00_000, 10_000)
        loan_amount   = st.slider("Loan Amount / yr (₹)", 10_000, 20_00_000, 2_00_000, 10_000)
        loan_tenure   = st.slider("Loan Tenure (yrs)", 1, 30, 5)
        cibil_rank    = st.slider("CIBIL Rank (1–10)", 1, 10, 7)

    with c3:
        cibil_score = st.slider("CIBIL Score", 300, 900, 750, 10)
        dpd         = st.slider("DPD (days)", 0, 1000, 0, 10)
        max_dpd     = st.slider("Max DPD (days)", 0, 1000, 0, 10)
        missed_emis = st.slider("Missed EMIs", 0, 60, 0)

    submitted = st.form_submit_button("Predict")

# ─────────────────────────────  prediction  ──────────────────────────────
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
        'CIBIL Score Ranking (850+=10, 800+=9, 750+=8, 700+=7, 650+=6, 600+=5, 550+=4, 450+=3, 400+=2, 300+=1)': cibil_rank,
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

    st.subheader("📈 Predicted Default Risk")
    fig, ax = plt.subplots(figsize=(16, .8))
    ax.barh([""], [probability], color=RISK_C)
    ax.barh([""], [1-probability], left=[probability], color=PRIMARY_C)
    ax.set_xlim(0,1)
    ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="black")
    ax.set_yticks([]); ax.set_facecolor(ACCENT_BG); fig.patch.set_facecolor(ACCENT_BG)
    ax.text(probability/2, 0, f"{probability:.0%}", va="center", ha="center", color="black", fontweight="bold")
    ax.tick_params(colors="black")
    st.pyplot(fig, use_container_width=True)

    # ─────────────  styled list layout  ─────────────
st.markdown("---")
st.subheader("🔍 Summary Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Key Inputs")
    st.markdown(f"""
    <ul style='list-style-type:none; padding-left:0; font-size:16px; line-height:1.6;'>
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
    <ul style='list-style-type:none; padding-left:0; font-size:16px; line-height:1.6;'>
        <li>📊 <b>EMI to Income Ratio:</b> {emi_to_income:.2f}</li>
        <li>💸 <b>Net Disposable Income:</b> ₹{net_disposable:,.2f}</li>
        <li>📅 <b>Missed EMI Rate:</b> {missed_emi_rate:.2f}</li>
        <li>📉 <b>Debt-to-Income (%):</b> {dti_after_loan_pct:.2f}%</li>
        <li>👨‍👩‍👧 <b>Surplus / Dependant / Month:</b> ₹{surplus_per_dep:,.2f}</li>
    </ul>
    """, unsafe_allow_html=True)
