import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.exception.exception import ScorePredictionException
from src.logging.logger import logger

from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

from src.utils.common import read_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(Path(SCHEMA_FILE_PATH))
        except Exception as e:
            raise ScorePredictionException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, index_col=0)
        except Exception as e:
            raise ScorePredictionException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config.columns)
            logger.info(f"Required number of columns:{number_of_columns}")
            logger.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise ScorePredictionException(e,sys)
        
    def validate_data_type(self, dataframe:pd.DataFrame) -> bool:
        try:
            numeric_features = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
            categorical_features = [col for col in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
            
            expected_numerical = set(self._schema_config.numerical_columns)
            expected_categorical = set(self._schema_config.categorical_columns)
            actual_numerical = set(numeric_features)
            actual_categorical = set(categorical_features)
            
            return actual_numerical == expected_numerical and actual_categorical == expected_categorical
        except Exception as e:
            raise ScorePredictionException(e,sys)
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            
            # validate number of columns
            train_columns_valid = self.validate_number_of_columns(train_dataframe)
            test_columns_valid = self.validate_number_of_columns(test_dataframe)

            # validate data type
            train_dtype_valid = self.validate_data_type(train_dataframe)
            test_dtype_valid = self.validate_data_type(test_dataframe)

            # global status: all the status needs to be true
            validation_status = all([
                train_columns_valid,
                test_columns_valid,
                train_dtype_valid,
                test_dtype_valid
            ]) 

            if not validation_status:
                logger.error("Data validation failed.")
                if not train_columns_valid:
                    logger.error("Train: Column count mismatch.")
                if not test_columns_valid:
                    logger.error("Test: Column count mismatch.")
                if not train_dtype_valid:
                    logger.error("Train: Data type mismatch.")
                if not test_dtype_valid:
                    logger.error("Test: Data type mismatch.")
            else:
                dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path,exist_ok=True)
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, index=False, header=True
                )
                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path, index=False, header=True
                )
                logger.info("Data validation successful. Valid datasets saved.")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
            )
            return data_validation_artifact

        except Exception as e:
            raise ScorePredictionException(e,sys)