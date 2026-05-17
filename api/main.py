# =========================
# api/main.py
# SenSante API - FastAPI
# =========================

# =========================
# IMPORTS
# =========================

import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

# =========================
# ENV
# =========================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# =========================
# APP FASTAPI
# =========================

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical Sénégal",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS & ENCODERS
# =========================

print("Chargement modèle...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print("Modèle chargé :", type(model).__name__)

# =========================
# SCHEMAS
# =========================

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)

    toux: bool
    fatigue: bool
    maux_tete: bool

    region: str


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str


class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

# =========================
# PROMPT LLM
# =========================

SYSTEM_PROMPT = """
Tu es un assistant médical sénégalais.
Explique simplement le diagnostic.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases.
Tu ne dois jamais poser de diagnostic.
"""

# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# PREDICTION (FIX WARNING SKLEARN)
# =========================

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # encodage
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message="Sexe invalide (M ou F)"
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message="Région inconnue"
        )

    # ✅ IMPORTANT : DataFrame (corrige warning sklearn)
    features = pd.DataFrame([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]], columns=feature_cols)

    # prédiction
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(np.max(probas))

    # confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # messages
    messages = {
        "paludisme": "Suspicion de paludisme. Consultez rapidement un médecin.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typhoide": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Aucune pathologie détectée."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )

# =========================
# EXPLAIN (GROQ LLM)
# =========================

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):

    if not groq_client:
        return ExplainOutput(
            explication="Service indisponible (GROQ_API_KEY manquante)."
        )

    prompt = (
        f"Patient: {data.sexe}, {data.age} ans, {data.region}\n"
        f"Température: {data.temperature}\n"
        f"Diagnostic: {data.diagnostic} ({data.probabilite:.0%})\n"
        f"Explique simplement."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur LLM: {str(e)}"

    return ExplainOutput(explication=explication)

# =========================
# MODEL INFO
# =========================

@app.get("/model-info")
def model_info():
    return {
        "model": type(model).__name__,
        "classes": list(model.classes_) if hasattr(model, "classes_") else [],
        "features": feature_cols
    }