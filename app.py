import streamlit as st 
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os, random

# ─────────────── PAGE SETUP ───────────────
st.set_page_config(page_title="Default Risk Predictor — NPA Probability", layout="wide")

# Light mode colors
ACCENT_BG = "#ffffff"
TEXT_COLOR = "#000000"
PRIMARY_C = "#1f77b4"
RISK_C    = "#d62728"
BAR_C     = "#00d491"

# Light theme override CSS
st.markdown(f"""
    <style>
    html, body, .stApp {{
        background-color: {ACCENT_BG};
        color: {TEXT_COLOR};
    }}
    .stSlider, .stSelectbox, .stNumberInput, label {{
        color: {TEXT_COLOR};
    }}
    </style>
""", unsafe_allow_html=True)

# ─────────────── LOAD MODEL ───────────────
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
st.sidebar.info("✅ Model loaded" if model else "⚠️ Model not found – using mock outputs")

# ─────────────── HEADER ───────────────
st.title("Bank Default Risk Predictor — NPA Probability")

# ─────────────── FORM ───────────────
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

# ─────────────── PREDICTION ───────────────
if submitted:
    emi_to_income      = (other_amis + loan_amount) / income if income else 0
    net_disposable     = income - other_amis - loan_amount
    missed_emi_rate    = missed_emis / loan_tenure if loan_tenure else 0
    dti_after_loan_pct = (other_amis + loan_amount) * 100 / income if income else 0
    surplus_per_dep    = (income - other_amis - loan_amount) / (12 * dependants) if dependants else 0

    row = {
        'Employment Type': employment_type,
        'Current Employment Years': current_employment_years,
        'Total Employment Years': total_employment_years,
        'SBI Customer': sbi_customer,
        'Location Type': location_type,
        'Income': income,
        'Other AMIs': other_amis,
        'Loan Amount': loan_amount,
        'Loan Tenure': loan_tenure,
        'CIBIL Rank': cibil_rank,
        'CIBIL Score': cibil_score,
        'DPD': dpd,
        'Max DPD': max_dpd,
        'Missed EMIs': missed_emis,
        'EMI to Income': emi_to_income,
        'Net Disposable Income': net_disposable,
        'Missed EMI Rate': missed_emi_rate,
        'Debt-to-Income Ratio': dti_after_loan_pct,
        'Surplus Per Dependant': surplus_per_dep
    }

    X = pd.DataFrame([row])
    probability = model.predict_proba(X)[0,1] if model else round(random.uniform(0.05, 0.85), 2)

    # ───────────── SLIM BAR ─────────────
    st.subheader("📈 Predicted Default Risk")
    fig, ax = plt.subplots(figsize=(16, .8))
    ax.barh([""], [probability], color=RISK_C)
    ax.barh([""], [1-probability], left=[probability], color=PRIMARY_C)
    ax.set_xlim(0,1)
    ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(["0%","25%","50%","75%","100%"], color=TEXT_COLOR)
    ax.set_yticks([])
    ax.set_facecolor(ACCENT_BG)
    fig.patch.set_facecolor(ACCENT_BG)
    ax.text(probability/2, 0, f"{probability:.0%}", va="center", ha="center", color="white", fontweight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    st.pyplot(fig, use_container_width=True)

    # ───────────── INPUT MINI-CHART ─────────────
    st.subheader("📊 Key Inputs (scaled)")
    summary_df = pd.DataFrame({
        "Parameter": ["Income", "Loan Amt", "Other AMIs", "CIBIL", "DPD", "Missed EMIs"],
        "Value":     [income, loan_amount, other_amis, cibil_score, dpd, missed_emis]
    })
    scaled = summary_df.Value / summary_df.Value.max()
    fig2, ax2 = plt.subplots(figsize=(8, 1.6))
    ax2.barh(summary_df.Parameter, scaled, color=BAR_C)
    for y, v, orig in zip(summary_df.Parameter, scaled, summary_df.Value):
        ax2.text(v+0.02, y, f"{orig:,}", va="center", color=TEXT_COLOR, fontsize=9)
    ax2.set_xlim(0,1)
    ax2.set_facecolor(ACCENT_BG)
    fig2.patch.set_facecolor(ACCENT_BG)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.invert_yaxis()
    for spine in ax2.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    st.pyplot(fig2, use_container_width=True)

    # ───────────── ENGINEERED MINI-CHART ─────────────
    st.subheader("🧮 Engineered Features")
    engineered_df = pd.DataFrame({
        "Feature": [
            "EMI to Income", 
            "Net Disposable Income", 
            "Missed EMI Rate", 
            "Debt-to-Income Ratio", 
            "Surplus Per Dependant"
        ],
        "Value": [
            emi_to_income, 
            net_disposable, 
            missed_emi_rate, 
            dti_after_loan_pct, 
            surplus_per_dep
        ]
    })
    scaled_e = engineered_df.Value / (engineered_df.Value.abs().max() or 1)
    fig3, ax3 = plt.subplots(figsize=(8, 1.6))
    ax3.barh(engineered_df.Feature, scaled_e, color=BAR_C)
    for y, v, orig in zip(engineered_df.Feature, scaled_e, engineered_df.Value):
        label = f"{orig:,.2f}" if abs(orig) > 1 else f"{orig:.3f}"
        ax3.text(v + 0.02, y, label, va="center", color=TEXT_COLOR, fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_facecolor(ACCENT_BG)
    fig3.patch.set_facecolor(ACCENT_BG)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.invert_yaxis()
    for spine in ax3.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    st.pyplot(fig3, use_container_width=True)
