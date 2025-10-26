import os
import sys
from datetime import datetime
from dataclasses import dataclass

from src import constants

date_format = "%m_%d_%Y_%H_%M_%S"


class TrainingPipelineConfig():
    def __init__(self, timestamp = datetime.now()):
        timestamp = timestamp.strftime(date_format)
        self.pipeline_name = constants.PIPELINE_NAME
        self.artifact_name = constants.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir=os.path.join("final_models")
        self.timestamp:str = timestamp

''' Old version for DataIngestionConfig()'''
# @dataclass
# class DataIngestionConfig():
#     train_data_path: str=os.path.join('artifacts',"train.csv")
#     test_data_path: str=os.path.join('artifacts',"test.csv")
#     raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestionConfig():
    def __init__(self, constants_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(
            constants_config.artifact_dir,
            constants.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                constants.DATA_INGESTION_FEATURE_STORE_DIR,
                constants.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir,
                constants.DATA_INGESTION_INGESTED_DIR,
                constants.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir,
                constants.DATA_INGESTION_INGESTED_DIR,
                constants.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = constants.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = constants.DATA_INGESTION_DATABASE_NAME
        
        
class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            constants.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            constants.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            constants.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            constants.TEST_FILE_NAME
        )