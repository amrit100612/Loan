from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Pipeline Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found in project folder")

model = pickle.load(open(MODEL_PATH, "rb"))

# -----------------------------
# Input Schema (All 11 Features)
# -----------------------------
class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

# -----------------------------
# Serve Frontend
# -----------------------------
@app.get("/")
def home():
    return FileResponse("index.html")

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: LoanInput):

    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

        # Use pipeline directly (NO manual encoding)
        prediction = model.predict(df)[0]

        # Convert result
        if prediction == "Y":
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}