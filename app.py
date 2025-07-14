import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os, random

st.set_page_config(layout="wide")
ACCENT_BG = "#111111"
PRIMARY_C = "#1f77b4"       
RISK_C    = "#d62728"      
BAR_C     = "#00d491"     


st.markdown(
    f"""
    <style>
    body, .reportview-container, .sidebar .sidebar-content {{
        background-color: {ACCENT_BG};
        color: white;
    }}
    h1, h2, h3, h4, h5, h6, label, p, .stSlider, .stSelectbox, .stNumberInput {{
        color: black;
    }}
    </style>""",
    unsafe_allow_html=True
)

#model load
MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
st.sidebar.info("Model loaded" if model else "Model not found – mock output")


st.title("Bank Default Risk Predictor — NPA Probability")

#form begins
with st.form("prediction_form"):
    st.header("Applicant Details")
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

#form ends
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

    st.subheader("Predicted Default Risk")
    fig, ax = plt.subplots(figsize=(16, .8))        
    ax.barh([""], [probability], color=RISK_C)
    ax.barh([""], [1-probability], left=[probability], color=PRIMARY_C)
    ax.set_xlim(0,1)
    ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="white")
    ax.set_yticks([]); ax.set_facecolor(ACCENT_BG); fig.patch.set_facecolor(ACCENT_BG)
    ax.text(probability/2, 0, f"{probability:.0%}", va="center", ha="center", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Key Inputs")
    summary_df = pd.DataFrame({
        "Parameter": ["Income","Loan Amt","Other AMIs","CIBIL","DPD","Missed EMIs"],
        "Value":     [income, loan_amount, other_amis, cibil_score, dpd, missed_emis]
    })


    scaled = summary_df.Value / summary_df.Value.max()
    fig2, ax2 = plt.subplots(figsize=(8,1.6))     
    ax2.barh(summary_df.Parameter, scaled, color=BAR_C)
    for y,v,orig in zip(summary_df.Parameter, scaled, summary_df.Value):
        ax2.text(v+0.02, y, f"{orig:,}", va="center", color="white", fontsize=9)
    ax2.set_xlim(0,1); ax2.set_facecolor(ACCENT_BG); fig2.patch.set_facecolor(ACCENT_BG)
    ax2.tick_params(colors="white"); ax2.invert_yaxis() 
    for spine in ax2.spines.values():
        spine.set_edgecolor("white")
    st.pyplot(fig2, use_container_width=True)

    st.subheader("Engineered Features")
    engineered_df = pd.DataFrame({
        "Feature": [
            "EMI to Income", 
            "Net Disposable Income", 
            "Missed EMI Rate", 
            "Debt-to-Income (%)", 
            "Surplus / Dependant / Mo"
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
        ax3.text(v + 0.02, y, label, va="center", color="white", fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_facecolor(ACCENT_BG)
    fig3.patch.set_facecolor(ACCENT_BG)
    ax3.tick_params(colors="white")
    ax3.invert_yaxis()
    for spine in ax3.spines.values():
        spine.set_edgecolor("white")
    st.pyplot(fig3, use_container_width=True)
