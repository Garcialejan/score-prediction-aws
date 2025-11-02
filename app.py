import os
import sys
from fastapi import FastAPI
import pymongo
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response, RedirectResponse

from src.exception.exception import ScorePredictionException
from src.logging.logger import logger
from src.pipeline.train_pipeline import TrainingPipeline

from src.utils.common import load_object
from src.utils.ml_models import ScoreModel

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
app = FastAPI()
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
    
if __name__ == "__main__":
    app_run(app, host="127.0.0.1", port=8080)