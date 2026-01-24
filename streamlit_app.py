import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Water Quality Prediction System",
    layout="centered"
)

# ---------------------------
# Title & System Description
# ---------------------------
st.title("💧 Water Quality & Infrastructure Prediction System")

st.markdown("""
This system provides **decision-support insights** for **water authorities, government agencies, and environmental management organizations**.

**Purpose of this system:**
- Assess overall **water quality condition** using the CCME Water Quality Index (WQI)
- Classify water quality status (e.g., Good, Fair, Marginal)
- Indicate potential **infrastructure anomaly risk** based on water conditions

⚠️ *This tool is intended for decision support and analysis, not as a replacement for regulatory compliance assessments.*
""")

st.divider()

# ---------------------------
# Input section
# ---------------------------
features = {}

st.subheader("🧪 Physicochemical Water Quality Parameters")
st.caption("Enter measured or estimated values. Hover over each field for guidance on meaning and acceptable ranges.")

features["Ammonia (mg/l)"] = st.number_input(
    "Ammonia (mg/L)",
    min_value=0.0,
    max_value=10.0,
    value=0.4,
    help="Ammonia indicates organic pollution and wastewater impact. Typical surface water values are usually below 1 mg/L."
)

features["Biochemical Oxygen Demand (mg/l)"] = st.number_input(
    "Biochemical Oxygen Demand – BOD (mg/L)",
    min_value=0.0,
    max_value=50.0,
    value=4.0,
    help="BOD measures oxygen demand from organic matter. Higher values indicate higher pollution levels."
)

features["Dissolved Oxygen (mg/l)"] = st.number_input(
    "Dissolved Oxygen – DO (mg/L)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    help="Dissolved oxygen supports aquatic life. Values below 5 mg/L may indicate ecological stress."
)

features["Orthophosphate (mg/l)"] = st.number_input(
    "Orthophosphate (mg/L)",
    min_value=0.0,
    max_value=5.0,
    value=0.15,
    help="Orthophosphate contributes to nutrient enrichment and eutrophication. Lower values are generally better."
)

features["pH (ph units)"] = st.number_input(
    "pH",
    min_value=0.0,
    max_value=14.0,
    value=7.0,
    help="pH indicates acidity or alkalinity. Natural surface waters typically range from 6.5 to 8.5."
)

features["Temperature (cel)"] = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=40.0,
    value=12.0,
    help="Water temperature affects oxygen solubility and biological activity."
)

features["Nitrogen (mg/l)"] = st.number_input(
    "Total Nitrogen (mg/L)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    help="Total nitrogen represents nutrient loading that can contribute to algal growth."
)

features["Nitrate (mg/l)"] = st.number_input(
    "Nitrate (mg/L)",
    min_value=0.0,
    max_value=50.0,
    value=4.5,
    help="Nitrate is a major nutrient pollutant. Elevated levels may indicate agricultural or wastewater runoff."
)

st.divider()

# ---------------------------
# Time & engineered features
# ---------------------------
st.subheader("🕒 Time & Contextual Information")

features["Quarter"] = st.selectbox(
    "Monitoring Quarter",
    [1, 2, 3, 4],
    help="Select the calendar quarter corresponding to the monitoring period."
)

# ---------------------------
# Engineered features (DO NOT CHANGE)
# ---------------------------
features["Ammonia_BOD"] = features["Ammonia (mg/l)"] * features["Biochemical Oxygen Demand (mg/l)"]
features["Orthophosphate_Temp"] = features["Orthophosphate (mg/l)"] * features["Temperature (cel)"]
features["Ammonia_log"] = 0.0
features["Orthophosphate_log"] = 0.0
features["BOD_scaled"] = 0.0

st.divider()

# ---------------------------
# Prediction button
# ---------------------------
if st.button("🔮 Run Prediction"):
    payload = {"features": features}

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed successfully.")

            st.subheader("📊 Prediction Results")

            st.markdown("""
            **How to read the results:**
            - The CCME Water Quality Index (WQI) summarizes overall water quality on a scale from 0 to 100.
            - Higher WQI values indicate better water quality.
            - The WQI Class provides a qualitative interpretation suitable for reporting and decision-making.
            """)

            st.write("**CCME Water Quality Index (XGBoost Model):**", round(result["CCME_Value_XGB"], 2))
            st.write("**CCME Water Quality Index (Neural Network Model):**", round(result["CCME_Value_NN"], 2))

            st.write("**Overall Water Quality Classification:**", result["CCME_WQI_Class"])

            st.markdown("""
            **Infrastructure Anomaly Risk:**
            - Represents the likelihood of abnormal infrastructure-related conditions inferred from water quality patterns.
            - Values closer to 1 indicate higher potential risk.
            """)

            st.write(
                "**Infrastructure Anomaly Risk Score:**",
                round(result["Infra_Anomaly_NN"], 3)
            )

            st.info(
                "ℹ️ This system is intended to support environmental monitoring, planning, and early risk identification. "
                "Final regulatory or operational decisions should consider field validation and expert review."
            )

        else:
            st.error(response.json().get("detail", "Prediction failed."))

    except Exception as e:
        st.error(str(e))
