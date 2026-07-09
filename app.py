import streamlit as st
import requests
import os


# 1. Page Configuration & Styling
st.set_page_config(page_title="Apex Credit Risk Intelligence", page_icon="💳", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #000000;}
    .stButton>button {background-color: #2e6fdf; color: white; border-radius: 8px; width: 100%;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 6px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

st.title("💳 Apex Credit Risk Intelligence Dashboard")
st.caption("Real-time credit default probability analysis powered by Gradient Boosting AI.")
st.write("---")

# 2. Setup Columns for Input vs Output
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Applicant Information")
    
    # User-friendly input elements grouped logically
    age = st.slider("Applicant Age", 18, 100, 35)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
    income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
    debt_ratio = st.number_input("Debt Ratio (Monthly Debt / Monthly Income)", min_value=0.0, max_value=10.0, value=0.3, step=0.05)
    
    st.write("**Credit History Details**")
    utilization = st.slider("Revolving Credit Utilization (Total Balance / Credit Limit)", 0.0, 2.0, 0.3, step=0.05)
    open_lines = st.number_input("Number of Open Credit Lines & Loans", min_value=0, value=5)
    real_estate = st.number_input("Number of Real Estate Loans", min_value=0, value=1)
    
    st.write("**Delinquency History**")
    late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, value=0)
    late_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)
    
    submit = st.button("Analyze Risk Profile")

with col2:
    st.subheader("📊 Risk Assessment Result")
    
    if submit:
        # Format input data into JSON payload
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "age": age,
            "NumberOfTime30_59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": float(income),
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60_89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": float(dependents)
        }
        
        try:
            # Send data to your FastAPI local endpoint 
            # (When deploying to Render later, change this URL to your live Render API URL!)
            API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
            response = requests.post(API_URL, json=payload)
            result = response.json()
            
            if result.get("status") == "success":
                data = result["data"]
                prob = data["probability_of_default"]
                decision = data["decision"]
                rating = data["risk_rating"]
                
                # Dynamic visual cues based on application outcome
                if decision == "Approved":
                    st.success(f"🎉 Application Status: {decision}")
                    accent_color = "green"
                else:
                    st.error(f"❌ Application Status: {decision}")
                    accent_color = "red"
                
                # Show key metrics in structured UI cards
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(label="Default Probability", value=f"{prob * 100:.2f}%")
                with m2:
                    st.metric(label="Risk Tier", value=rating)
                
                # Display a visual progress bar indicating risk level
                st.write("**Risk Spectrum:**")
                st.progress(prob)
                
            else:
                st.error("Error processing credit prediction response.")
        except Exception as e:
            st.error(f"Could not connect to backend server. Make sure FastAPI is running! Error: {e}")
    else:
        st.info("Adjust values on the left pane and click 'Analyze Risk Profile' to generate an assessment.")