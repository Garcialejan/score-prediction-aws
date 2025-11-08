import os
import sys
from fastapi import FastAPI
import pymongo
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import request

from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response, RedirectResponse

from src.exception.exception import ScorePredictionException
from src.logging.logger import logger
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictPipeline, CustomData



import certifi
ca = certifi.where() 

# Create the connection to mongodb
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("db_username")
password = os.getenv("db_password")
if not username or not password:
    raise ScorePredictionException("Database credentials are missing.", sys)
mongodb_uri = f"mongodb+srv://{username}:{password}@scoringproject.lcjwpi4.mongodb.net/?appName=ScoringProject"

client = pymongo.MongoClient(mongodb_uri, tlsCAFile=ca)

from src.constants import DATA_INGESTION_COLLECTION_NAME
from src.constants import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Start the FastAPI app creation
application = FastAPI()
app = application
origins = ["*"]

app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise ScorePredictionException(e,sys)
    

@app.post("/predict", response_class = Response)
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        table_html = results.to_html(classes='table table-striped')
        
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise ScorePredictionException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)