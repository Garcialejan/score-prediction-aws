import os
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from xgboost import XGBModel
from sklearn.metrics import r2_score

from src.exception.exception import ScorePredictionException
from src.logging.logger import logger
from src.utils.common import load_object, save_object
from src.utils.ml_models import evaluate_models, get_regresion_score

# import mlflow
# from urllib.parse import urlparse