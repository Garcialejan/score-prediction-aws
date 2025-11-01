import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logging.logger import logger
from src.utils.common import save_numpy_array_data, save_object
from src.exception.exception import ScorePredictionException

from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.constants import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except  Exception as e:
            raise ScorePredictionException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path, index_col=0)
        except Exception as e:
            raise ScorePredictionException(e,sys)
        
    @staticmethod
    def get_column_types(df: pd.DataFrame):
        """Return two lists: names of the columns with categorical or numerical data"""
        numerical_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        return numerical_columns, categorical_columns
        
    def get_data_transformer_object(self, numerical_columns: list, categorical_columns: list) -> Pipeline:
        """
        Function responsible for the data trnasformation

        Args:
          cls: DataTransformation

        Returns:
          A Pipeline object
        """
        try:
            num_pipeline= Pipeline(
                steps=[
                    ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise ScorePredictionException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logger.info("Strating data trasnformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            numerical_columns, categorical_columns = self.get_column_types(train_df)
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")
            
            ## training dataframe
            train_df['total_score'] = train_df[numerical_columns].sum(axis=1)
            train_df['average'] = train_df['total_score']/3
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN, "total_score"], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            ## testing dataframe
            test_df['total_score'] = test_df[numerical_columns].sum(axis=1)
            test_df['average'] = test_df['total_score']/3
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN, "total_score"], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            # create the pipeline to impute the data
            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            
            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            
            #preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
            
        except Exception as e:
            raise ScorePredictionException(e, sys)