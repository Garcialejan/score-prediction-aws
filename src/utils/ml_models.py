import sys
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception.exception import ScorePredictionException
from sklearn.model_selection import GridSearchCV
from src.entity.artifact_entity import RegresionMetricArtifact

class ScoreModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise ScorePredictionException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise ScorePredictionException(e,sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name in models.keys():
            model = models[model_name]
            para = param[model_name]

            # Hyperparameter searching
            gs = GridSearchCV(model, para, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_train, y_train)

            # take best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # train metrics
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)

            # test metrics
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)

            report[model_name] = {
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "train_mae": train_mae,
                "best_params": gs.best_params_
            }
            
        return report
    except Exception as e:
        raise ScorePredictionException(e, sys)
    
def get_regresion_score(y_true,y_pred) -> RegresionMetricArtifact:
    try:
        model_r2 = r2_score(y_true, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        model_mae = mean_absolute_error(y_true, y_pred)

        regresion_metric =  RegresionMetricArtifact(
            r2 = model_r2,
            rmse=model_rmse,
            mae=model_mae,
            )
        return regresion_metric
    except Exception as e:
        raise ScorePredictionException(e,sys)