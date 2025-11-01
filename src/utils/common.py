import os,sys
import yaml
import json
import pickle
import joblib
from pathlib import Path
import numpy as np
from typing import Any
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError

from src.exception.exception import ScorePredictionException
from src.logging.logger import logger

# Ensure library is designed to simplify testing and validation 
# of function arguments, return values, and other aspects. Provides
# decorators and helper functions to #*enforce type annotations,
#* constraints, or conditions

# The ConfigBox package is a python functionality that allows you to access 
# dictionary keys as if they were attributes It simplifies working
# with nested dictionaries. #* Simplifies working with JSON or YAML data

@ensure_annotations
def read_yaml_file(path_to_yaml: Path) -> ConfigBox:
    """
    Function to read yaml file and returns
    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file) # Read the yaml file
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content) # To use dict as arguments
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ScorePredictionException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ScorePredictionException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ScorePredictionException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise ScorePredictionException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ScorePredictionException(e, sys) from e