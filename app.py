import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os, random

st.set_page_config(layout="wide", page_title="Default Risk Predictor")
ACCENT_BG = "#0d1117"
PRIMARY_C = "#1f77b4"
RISK_C    = "#d62728"
BAR_C     = "#00d491"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {ACCENT_BG};
        color: white;
    }}
    .stApp {{
        background-color: {ACCENT_BG};
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: white;
    }}
    .stSlider > div {{
        color: #ddd;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .stForm button {{
        background-color: {BAR_C};
        color: black;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True
)

MODEL_PATH = "xgb_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
st.sidebar.success("✅ Model loaded" if model else "⚠️ Model not found")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


st.markdown(
    "<div style='text-align: center; font-weight:bold; font-size:40px; color: #444'>"
    "Bank Default Risk Predictor"
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center; font-size:18px; color: #444'>"
    "Estimate probability of loan default using applicant details"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("👤 Applicant Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        employment_type = st.selectbox("Employment Type", ["Govt (4)", "Salaried (3)", "Self-Employed (2)", "Pensioner (1)", "Other (0)", "Unemployed (5)"], index=1)
        employment_type = int(employment_type[-2])
        current_employment_years = st.slider("Current Employment Length (yrs)", 0, 40, 5)
        total_employment_years   = st.slider("Total Employment (yrs)", 0, 40, 10)
        sbi_customer             = st.radio("Existing SBI Customer?", ["No", "Yes"], horizontal=True) == "Yes"
        location_type            = st.selectbox("Location Type", ["Rural (1)", "Semi-Urban (2)", "Urban (3)"], index=2)
        location_type = int(location_type[-2])
        dependants               = st.slider("Number of Dependants", 1, 10, 2)

    with c2:
        income        = st.slider("Annual Income (₹)", 1_00_000, 50_00_000, 5_00_000, 50_000)
        other_amis    = st.slider("Other Loan AMIs (₹)", 0, 20_00_000, 1_00_000, 10_000)
        loan_amount   = st.slider("Requested Loan Amount /yr (₹)", 10_000, 20_00_000, 2_00_000, 10_000)
        loan_tenure   = st.slider("Loan Tenure (yrs)", 1, 30, 5)
        cibil_rank    = st.slider("CIBIL Rank (1–10)", 1, 10, 7)

    with c3:
        cibil_score = st.slider("CIBIL Score", 300, 900, 750, 10)
        dpd         = st.slider("DPD (days)", 0, 1000, 0, 10)
        max_dpd     = st.slider("Max DPD (days)", 0, 1000, 0, 10)
        missed_emis = st.slider("Missed EMIs", 0, 60, 0)

    submitted = st.form_submit_button("🔮 Predict Default Risk")

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
    probability = model.predict_proba(X)[0, 1] if model else round(random.uniform(0.05, 0.85), 2)

    st.markdown("---")
    st.subheader("Predicted Default Probability")
    fig, ax = plt.subplots(figsize=(16, .8))        
    ax.barh([""], [probability], color=RISK_C)
    ax.barh([""], [1 - probability], left=[probability], color=PRIMARY_C)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="white")
    ax.set_yticks([]); ax.set_facecolor(ACCENT_BG); fig.patch.set_facecolor(ACCENT_BG)
    ax.text(probability / 2, 0, f"{probability:.0%}", va="center", ha="center", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    st.pyplot(fig, use_container_width=True)

st.markdown("Income Allocation with Total Yearly Loan Amount")


loan_share = loan_amount / income if income else 0
amis_share = other_amis / income if income else 0
surplus_share = max(0, 1 - loan_share - amis_share)

sizes = [loan_share, amis_share, surplus_share]
labels = [
    f"Loan: ₹{loan_amount:,.0f} ({loan_share*100:.1f}%)",
    f"Yearly Loan Payments: ₹{other_amis:,.0f} ({amis_share*100:.1f}%)",
    f"Surplus: ₹{surplus_share * income:,.0f} ({surplus_share*100:.1f}%)"
]
inner_colors = [RISK_C, BAR_C, PRIMARY_C]
outer_colors = [ACCENT_BG]

total_loan_related = loan_amount + other_amis
total_loan_label = f"Total Loan Amt Post Loan\n₹{total_loan_related:,.0f}"

fig, ax = plt.subplots(figsize=(6, 6))

# Outer ring
ax.pie(
    [1],
    radius=1.1,
    labels=["Total Income"],
    colors=outer_colors,
    labeldistance=1.1,
    textprops={'fontsize': 12, 'color': 'white', 'fontweight': 'bold'},
    wedgeprops=dict(width=0.15, edgecolor='white')
)

# Inner ring
ax.pie(
    sizes,
    radius=0.95,
    labels=labels,
    colors=inner_colors,
    labeldistance=1.25,
    startangle=90,
    textprops={'fontsize': 11, 'color': 'white', 'fontweight': 'bold'},
    wedgeprops=dict(width=0.3, edgecolor='white')
)

ax.text(0, 0, total_loan_label,
        ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')

ax.set(aspect="equal")
fig.patch.set_facecolor(ACCENT_BG)

st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("Other Applicant Metrics")

other_metrics_df = pd.DataFrame({
    "Metric": ["Co-applicant Income", "Guarantor Income", "Dependants", "Applicant Age"],
    "Value": [coapplicant_income, guarantor_income, dependants, age]
})

st.dataframe(
    other_metrics_df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ]).set_properties(**{'text-align': 'center'}),
    use_container_width=False,
    height=200
)


