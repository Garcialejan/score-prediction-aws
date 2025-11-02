import os
import sys
import pandas as pd
import numpy as np


"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "average"
PIPELINE_NAME: str = "ScoringPrediction"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "stud.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
DATA_FILE_PATH = os.path.join("Scoring_Data", FILE_NAME)

SAVED_MODEL_DIR =os.path.join("final_models")
MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "ScoringData"
DATA_INGESTION_DATABASE_NAME: str = "ScoringProject"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"

# """
# Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
# """
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 5,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_R2: float = 0.7
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
TRAINING_BUCKET_NAME = "scorepredictiongarcialejan"

## MODEL TRAINING PARAMs
RANDOM_FOREST_PARAMS: dict = {
    'n_estimators': [32, 64, 128, 256],
    'criterion':['squared_error', 'friedman_mse'],
    'max_features':['sqrt','log2', None],
}

GRADIENT_BOOSTING_PARAMS: dict = {
    'loss':['squared_error', 'huber'],
    'learning_rate':[.1, .01, .05, .001],
    'n_estimators': [32, 64, 128, 256],
    'subsample':[0.6, 0.7, 0.8, 0.9, 1],
    'max_features':['sqrt','log2', None],
    # 'criterion':['squared_error', 'friedman_mse'],
}

XGB_PARAMS: dict = {
    'learning_rate':[.1, .01, .05, .001],
    'n_estimators': [32, 64, 128, 256],
    'subsample':[0.6, 0.7, 0.8, 0.9, 1],
    'reg_alpha': [0.2, 0.5, 0.7, 1],
    'reg_lambda': [1, 5, 10]
}

CATBOOSTING_PARAMS: dict = {
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [30, 50, 100]
}

ADABOOST_PARAMS: dict = {
    'n_estimators': [32, 50, 64, 128, 256],
    'learning_rate':[.1, .01, .05, .001],
    # 'loss':['linear','square','exponential'],
}

KNEIGHBORS_PARAMS: dict = {
    'n_neighbors': [3, 5, 10],
}

ELASTICNET_PARAMS: dict = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.2, 0.5, 0.7, 1]
}
