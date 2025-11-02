import os
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from xgboost import XGBModel

from src.exception.exception import ScorePredictionException
from src.logging.logger import logger
from src.utils.common import load_object, save_object, load_numpy_array_data
from src.utils.ml_models import evaluate_models, get_regresion_score, ScoreModel

from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact 
from src.entity.config_entity import ModelTrainerConfig
from src.constants import RANDOM_FOREST_PARAMS, GRADIENT_BOOSTING_PARAMS, XGB_PARAMS,\
                          CATBOOSTING_PARAMS, ADABOOST_PARAMS, KNEIGHBORS_PARAMS, ELASTICNET_PARAMS

class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise ScorePredictionException(e, sys)
            
    def train_model(self, X_train, y_train, X_test, y_test):
        '''
        Function to train the model with hyperparameter tunning
        and create the model trainer artifact for the predictions.
        '''
        models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "ElasticNet Regressor": ElasticNet()
            }
        
        params={
            "Random Forest": RANDOM_FOREST_PARAMS,
            "Gradient Boosting": GRADIENT_BOOSTING_PARAMS,
            "XGBRegressor": XGB_PARAMS,
            "CatBoosting Regressor": CATBOOSTING_PARAMS,
            "AdaBoost Regressor": ADABOOST_PARAMS,
            "KNeighbors Regressor": KNEIGHBORS_PARAMS,
            "ElasticNet Regressor": ELASTICNET_PARAMS
        }
        
        # We use GridSearch to find the best model with the best hyperparameters
        model_report = evaluate_models(X_train = X_train, y_train = y_train,
                                       X_test = X_test, y_test = y_test,
                                       models = models, param = params)
        
        ## To get best model name from dict
        best_model_name = max(model_report.keys(), key=lambda name: model_report[name]["test_rmse"])
        ## To get best model score from dict
        best_model_score = model_report[best_model_name]["test_rmse"]
        
        ## To get best model params from dict
        best_params = model_report[best_model_name]["best_params"]
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best score: {best_model_score:.3f}")
        
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)
        regresion_train_metric = get_regresion_score(y_true = y_train, y_pred = y_train_pred)
        
        y_test_pred = best_model.predict(X_test)
        regresion_test_metric = get_regresion_score(y_true = y_test, y_pred = y_test_pred)
        
        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok = True)

        Score_Model = ScoreModel(preprocessor = preprocessor, model = best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj = Score_Model)
            
        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path = self.model_trainer_config.trained_model_file_path,
            train_metric_artifact = regresion_train_metric,
            test_metric_artifact = regresion_test_metric
            )
        
        logger.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # loading train and test arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            
            return model_trainer_artifact
        except Exception as e:
            raise ScorePredictionException(e, sys)