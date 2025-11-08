import sys
import os
import pandas as pd
import pickle
import boto3
from src.exception.exception import ScorePredictionException
from src.logging.logger import logger
from src.utils.ml_models import ScoreModel

from src.constants import TRAINING_BUCKET_NAME, MODEL_TRAINER_TRAINED_MODEL_NAME, PREPROCESSING_OBJECT_FILE_NAME

class PredictPipeline:
    def __init__(self):
        pass
    
    @staticmethod
    def load_object_from_s3(s3_key: str):
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=TRAINING_BUCKET_NAME, Key=s3_key)
        body = response['Body'].read()
        return pickle.loads(body)

    def predict(self, x):
        try:
            preprocesor = self.load_object_from_s3(f"s3://{TRAINING_BUCKET_NAME}/final_model/{PREPROCESSING_OBJECT_FILE_NAME}")
            final_model = self.load_object_from_s3(f"s3://{TRAINING_BUCKET_NAME}/final_model/{MODEL_TRAINER_TRAINED_MODEL_NAME}")
            model = ScoreModel(preprocessor = preprocesor, model = final_model)
            preds = model.predict(x)
            return preds
        except Exception as e:
            raise ScorePredictionException(e,sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise ScorePredictionException(e, sys)
