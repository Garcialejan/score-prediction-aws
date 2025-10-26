from src.components.data_ingestion import DataIngestion
# from src.components.data_validation import DataValidation
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
from src.exception.exception import ScorePredictionException
from src.logging.logger import logger

from src.entity.config_entity import DataIngestionConfig #, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        training_pipeline_config=TrainingPipelineConfig()
        
        # Data Ingestion
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logger.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed")
        print(data_ingestion_artifact)
        
    except Exception as e:
           raise ScorePredictionException(e,sys)