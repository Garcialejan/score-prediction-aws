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