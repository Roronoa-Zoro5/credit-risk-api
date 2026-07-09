# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# from typing import Optional
# import joblib
# import numpy as np

# # 1. Initialize the FastAPI application
# app = FastAPI(
#     title="Credit Risk Assessment API",
#     description="API to predict default probability using Kaggle's GiveMeSomeCredit dataset features.",
#     version="1.0.0"
# )

# # 2. Load our exported artifacts 
# # (Since main.py is in the root and artifacts is a folder next to it, we don't use '..')
# # MODEL = joblib.load("Notebooks/artifacts/gb_model.joblib")

# import xgboost as xgb

# MODEL = xgb.XGBClassifier()
# MODEL.load_model("Notebooks/artifacts/xgb_model.json")
# CONSTANTS = joblib.load("Notebooks/artifacts/preprocessing_constants.joblib")

# # 3. Define the Input Data Schema
# # This ensures whoever sends data to your API uses the exact Kaggle columns.
# class ApplicantData(BaseModel):
#     RevolvingUtilizationOfUnsecuredLines: float
#     age: int
#     NumberOfTime30_59DaysPastDueNotWorse: int
#     DebtRatio: float
#     MonthlyIncome: Optional[float] = Field(default=None, description="Monthly income. Can be null.")
#     NumberOfOpenCreditLinesAndLoans: int
#     NumberOfTimes90DaysLate: int
#     NumberRealEstateLoansOrLines: int
#     NumberOfTime60_89DaysPastDueNotWorse: int
#     NumberOfDependents: Optional[float] = Field(default=None, description="Number of dependents. Can be null.")

# # 4. Create the Prediction Endpoint
# @app.post("/predict")
# def predict_risk(applicant: ApplicantData):
    
#     # --- HANDLING MISSING VALUES ---
#     # If the web request leaves MonthlyIncome or Dependents blank (None), 
#     # we fall back to the training medians we exported in Step 1.
#     income = applicant.MonthlyIncome
#     if income is None or np.isnan(income):
#         income = CONSTANTS["monthly_income_median"]
        
#     dependents = applicant.NumberOfDependents
#     if dependents is None or np.isnan(dependents):
#         dependents = CONSTANTS["dependents_median"]
        
#     # Construct the array in the EXACT column sequence your Gradient Booster expects
#     features = np.array([[
#         applicant.RevolvingUtilizationOfUnsecuredLines,
#         applicant.age,
#         applicant.NumberOfTime30_59DaysPastDueNotWorse,
#         applicant.DebtRatio,
#         income,
#         applicant.NumberOfOpenCreditLinesAndLoans,
#         applicant.NumberOfTimes90DaysLate,
#         applicant.NumberRealEstateLoansOrLines,
#         applicant.NumberOfTime60_89DaysPastDueNotWorse,
#         dependents
#     ]])
    
#     # --- INFERENCE ---
#     # Get the raw probability of defaulting [Prob(0), Prob(1)]
#     probabilities = MODEL.predict_proba(features)[0]
#     probability_of_default = probabilities[1]
    
#     # --- BUSINESS LOGIC ---
#     # We lower the default threshold from 0.5 to 0.25 to actively fix that low Recall 
#     # you noticed in your classification report.
#     BUSINESS_THRESHOLD = 0.25
#     if probability_of_default >= BUSINESS_THRESHOLD:
#         decision = "Rejected"
#         risk_rating = "High Risk"
#     else:
#         decision = "Approved"
#         risk_rating = "Low Risk"
        
#     # Return a clean JSON response back to the client
#     return {
#         "status": "success",
#         "data": {
#             "probability_of_default": round(float(probability_of_default), 4),
#             "decision": decision,
#             "risk_rating": risk_rating
#         }
#     }

# @app.get("/")
# def root():
#     return {"message": "Credit Risk API is live. Go to /docs for the interactive UI."}


from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import xgboost as xgb
import joblib
import numpy as np

app = FastAPI(
    title="Credit Risk Assessment API",
    description="API to predict default probability using Kaggle's GiveMeSomeCredit dataset features.",
    version="1.0.0"
)

MODEL = xgb.XGBClassifier()
MODEL.load_model("Notebooks/artifacts/xgb_model.json")
CONSTANTS = joblib.load("Notebooks/artifacts/preprocessing_constants.joblib")

class ApplicantData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: Optional[float] = Field(default=None)
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: Optional[float] = Field(default=None)

@app.post("/predict")
def predict_risk(applicant: ApplicantData):

    income = applicant.MonthlyIncome
    if income is None or np.isnan(income):
        income = CONSTANTS["monthly_income_median"]

    dependents = applicant.NumberOfDependents
    if dependents is None or np.isnan(dependents):
        dependents = CONSTANTS["dependents_median"]

    late_30_59 = applicant.NumberOfTime30_59DaysPastDueNotWorse
    late_60_89 = applicant.NumberOfTime60_89DaysPastDueNotWorse
    late_90 = applicant.NumberOfTimes90DaysLate

    # --- replicate worse_deliq() ---
    if late_90 > 0:
        worsedeliq = 3
    elif late_30_59 > 0:
        worsedeliq = 2
    elif late_60_89 > 0:
        worsedeliq = 1
    else:
        worsedeliq = 0

    # --- replicate total_time_late ---
    total_time_late = late_30_59 + late_90 + late_60_89

    # --- replicate total_loans ---
    total_loans = applicant.NumberOfOpenCreditLinesAndLoans + applicant.NumberRealEstateLoansOrLines

    # --- replicate debtratio_extreme_flag ---
    debtratio_extreme_flag = int(applicant.DebtRatio > 10)

    features = np.array([[
        applicant.RevolvingUtilizationOfUnsecuredLines,
        applicant.age,
        late_30_59,
        applicant.DebtRatio,
        income,
        applicant.NumberOfOpenCreditLinesAndLoans,
        late_90,
        applicant.NumberRealEstateLoansOrLines,
        late_60_89,
        dependents,
        worsedeliq,
        total_time_late,
        total_loans,
        debtratio_extreme_flag
    ]])

    probabilities = MODEL.predict_proba(features)[0]
    probability_of_default = probabilities[1]

    BUSINESS_THRESHOLD = 0.25
    if probability_of_default >= BUSINESS_THRESHOLD:
        decision = "Rejected"
        risk_rating = "High Risk"
    else:
        decision = "Approved"
        risk_rating = "Low Risk"

    return {
        "status": "success",
        "data": {
            "probability_of_default": round(float(probability_of_default), 4),
            "decision": decision,
            "risk_rating": risk_rating
        }
    }

@app.get("/")
def root():
    return {"message": "Credit Risk API is live. Go to /docs for the interactive UI."}