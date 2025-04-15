from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the saved models
logging.info("Loading models...")
with open("./models/clustering_model.pkl", "rb") as file:
    model_data = pickle.load(file)

law_gov_model = joblib.load('./models/law_gov_model.pkl')
media_model = joblib.load('./models/media_communication.pkl')
knn_model = joblib.load('./models/knn_tech_model.pkl')

with open("./models/gmm.pkl", "rb") as file:
    model_bundle = pickle.load(file)

scaler = model_bundle["scaler"]
pca = model_bundle["pca"]
gmm_model = model_bundle["model"]
cluster_labels = model_bundle["cluster_labels"]

pipeline = model_data["pipeline"]
labels_map = model_data["labels_map"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TechSuitabilityScores(BaseModel):
    Math_Score: float
    Science_Score: float
    Logical_Thinking: float
    Problem_Solving_Skills: float
    Programming_Knowledge: float
    Coding_Experience: float
    Tech_Exposure: float
    Interest_in_Tech: float
    Participation_in_Coding_Clubs: float
    Participation_in_Science_Fairs: float
    STEM_Activities: float
    Attention_to_Detail: float
    Project_Work_Experience: float
    Communication_Skills: float
    Time_Management: float
    Adaptability: float
    Digital_Literacy: float

@app.post("/predict/tech")
def predict_tech_suitability(scores: TechSuitabilityScores):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([scores.dict()])

        # Predict suitability score
        prediction = knn_model.predict(input_data)[0]
        return {"suitability_score": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error in /predict/tech-suitability: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Define input models
class GmmData(BaseModel):
    O_score: float
    C_score: float
    E_score: float
    A_score: float
    N_score: float
    Numerical_Aptitude: float
    Spatial_Aptitude: float
    Perceptual_Aptitude: float
    Abstract_Reasoning: float
    Verbal_Reasoning: float

class StudentSkills(BaseModel):
    Social_Studies_Score: float
    Language_Score: float
    Critical_Thinking: float
    Debate_Skills: float
    Public_Speaking: float
    Leadership: float
    Research_Skills: float
    Communication_Skills: float
    Time_Management: float
    Adaptability: float

class AptitudeScores(BaseModel):
    Numerical_Aptitude: float
    Spatial_Aptitude: float
    Perceptual_Aptitude: float
    Abstract_Reasoning: float
    Verbal_Reasoning: float

# Endpoints
@app.post("/predict/gmm")
async def predict_gmm(data: GmmData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Preprocess data
        scaled_data = scaler.transform(input_df)
        pca_data = pca.transform(scaled_data)

        # Predict cluster
        cluster = int(gmm_model.predict(pca_data)[0])
        competency_level = cluster_labels.get(cluster, "Unknown")
        probabilities = gmm_model.predict_proba(pca_data)[0].tolist()

        return {
            "cluster": cluster,
            "competency_level": competency_level,
            "probabilities": probabilities
        }
    except Exception as e:
        logging.error(f"Error in /predict/gmm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/media")
def predict_media(data: StudentSkills):
    try:
        input_data = np.array([list(data.dict().values())])
        prediction = media_model.predict(input_data)[0]
        return {"media_communication": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error in /predict/media: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/law-gov")
def predict_law_gov(data: StudentSkills):
    try:
        input_data = np.array([list(data.dict().values())])
        prediction = law_gov_model.predict(input_data)[0]
        return {"law_and_government": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error in /predict/law-gov: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/aptitude")
def predict_aptitude(scores: AptitudeScores):
    try:
        input_data = [[
            scores.Numerical_Aptitude,
            scores.Spatial_Aptitude,
            scores.Perceptual_Aptitude,
            scores.Abstract_Reasoning,
            scores.Verbal_Reasoning
        ]]
        cluster = pipeline.predict(input_data)[0]
        competency_level = labels_map.get(cluster, "Unknown")
        return {"competency_level": competency_level}
    except Exception as e:
        logging.error(f"Error in /predict/aptitude: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)