"""
Startup Funding Predictor
--------------------------
A Streamlit app that serves a pre-trained K-Nearest Neighbors Regressor
(wrapped in a StandardScaler pipeline) to predict startup funding amount
(USD) based on company attributes.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Startup Funding Predictor",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

# Exact feature order the model was trained on.
FEATURE_NAMES = [
    "Funding_Year", "Employees",
    "Industry_AgriTech", "Industry_CleanTech", "Industry_Cybersecurity",
    "Industry_E-commerce", "Industry_EdTech", "Industry_FinTech",
    "Industry_HealthTech", "Industry_Logistics", "Industry_SaaS",
    "City_Bengaluru", "City_Chennai", "City_Delhi", "City_Hyderabad",
    "City_Jaipur", "City_Kolkata", "City_Mumbai", "City_Noida", "City_Pune",
    "State_Gujarat", "State_Karnataka", "State_Maharashtra", "State_Rajasthan",
    "State_Tamil Nadu", "State_Telangana", "State_Uttar Pradesh", "State_West Bengal",
    "Funding_Stage_Seed", "Funding_Stage_Series A", "Funding_Stage_Series B",
    "Funding_Stage_Series C",
    "Lead_Investor_Blume Ventures", "Lead_Investor_Kalaari Capital",
    "Lead_Investor_Matrix Partners", "Lead_Investor_Nexus Venture Partners",
    "Lead_Investor_Peak XV", "Lead_Investor_Sequoia Capital",
    "Lead_Investor_Tiger Global",
]

INDUSTRIES = ["AgriTech", "CleanTech", "Cybersecurity", "E-commerce", "EdTech",
              "FinTech", "HealthTech", "Logistics", "SaaS"]
CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur",
          "Kolkata", "Mumbai", "Noida", "Pune"]
STATES = ["Gujarat", "Karnataka", "Maharashtra", "Rajasthan",
          "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"]
FUNDING_STAGES = ["Seed", "Series A", "Series B", "Series C"]
LEAD_INVESTORS = ["Blume Ventures", "Kalaari Capital", "Matrix Partners",
                   "Nexus Venture Partners", "Peak XV", "Sequoia Capital",
                   "Tiger Global"]


# --------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    here = Path(__file__).parent
    candidate_paths = [MODEL_PATH, Path("model.pkl")]
    if here.exists():
        candidate_paths += list(here.glob("*.pkl"))

    for path in candidate_paths:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)

    files_here = [p.name for p in here.iterdir()] if here.exists() else []
    st.error(
        "**model.pkl not found.**\n\n"
        f"Files actually present next to app.py: {files_here}\n\n"
        "Make sure `model.pkl` is committed to your GitHub repo in the same "
        "folder as `app.py`."
    )
    st.stop()


def build_feature_row(funding_year, employees, industry, city,
                       state, funding_stage, lead_investor):
    """Assemble a single-row DataFrame matching the model's expected columns."""
    row = {name: 0 for name in FEATURE_NAMES}

    row["Funding_Year"] = funding_year
    row["Employees"] = employees

    row[f"Industry_{industry}"] = 1
    row[f"City_{city}"] = 1
    row[f"State_{state}"] = 1
    row[f"Funding_Stage_{funding_stage}"] = 1
    row[f"Lead_Investor_{lead_investor}"] = 1

    return pd.DataFrame([row], columns=FEATURE_NAMES)


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
st.sidebar.header("📋 Startup Details")

funding_year = st.sidebar.slider("Funding Year", min_value=2018, max_value=2026, value=2024)
employees = st.sidebar.number_input("Number of Employees", min_value=1, max_value=10000, value=50, step=1)
industry = st.sidebar.selectbox("Industry", INDUSTRIES)
city = st.sidebar.selectbox("City", CITIES)
state = st.sidebar.selectbox("State", STATES)
funding_stage = st.sidebar.selectbox("Funding Stage", FUNDING_STAGES)
lead_investor = st.sidebar.selectbox("Lead Investor", LEAD_INVESTORS)

predict_clicked = st.sidebar.button("🔮 Predict Funding Amount", use_container_width=True)

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title("💸 Startup Funding Predictor")
st.markdown(
    "Estimate a startup's likely **funding amount (USD)** using a "
    "K-Nearest Neighbors model trained on Indian startup funding data. "
    "Fill in the details in the sidebar and click **Predict**."
)

st.divider()

with st.expander("ℹ️ About this model"):
    st.markdown(
        """
        - **Algorithm:** K-Nearest Neighbors Regressor (k = 5, distance-weighted),
          with feature scaling
        - **Trained on:** 240 startup funding records
        - **Predicts:** Funding amount in USD, based on funding year, employee
          count, industry, city, state, funding stage, and lead investor

        **Note on accuracy:** In this dataset, funding amount shows very little
        real statistical relationship with these attributes (cross-validated
        R² ≈ 0), so predictions will cluster fairly close to the dataset
        average (~$24M) with only modest movement based on input. This is an
        honest reflection of the data, not a bug — a stronger model would need
        richer / more informative training data.
        """
    )

if predict_clicked:
    model = load_model()
    X = build_feature_row(
        funding_year, employees, industry, city,
        state, funding_stage, lead_investor
    )

    with st.spinner("Predicting..."):
        prediction = model.predict(X)[0]

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Funding Amount", f"${prediction:,.0f}")
    with col2:
        st.metric("Approx. in ₹ Crore (@ ₹83/USD)", f"₹{(prediction * 83) / 1e7:,.2f} Cr")

    with st.expander("🔍 View input feature vector"):
        st.dataframe(X.T.rename(columns={0: "Value"}), use_container_width=True)

else:
    st.info("Set the startup details in the sidebar, then click **Predict Funding Amount**.")

st.divider()
st.caption("Built with Streamlit · Model: scikit-learn KNeighborsRegressor (StandardScaler + KNN pipeline)")
