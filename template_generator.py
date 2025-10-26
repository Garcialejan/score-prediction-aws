import os
from pathlib import Path
import logging

project_name = "src"
logging.basicConfig(level = logging.INFO,
                    format  ='[%(asctime)s]: %(message)s:')

list_of_files = [
    f"{project_name}/__init__.py",
    
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/model_trainer.py",
    
    f"{project_name}/logging/__init__.py",
    
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/train_pipeline.py",
    f"{project_name}/pipeline/predict_pipeline.py",
    
    f"{project_name}/utils/__init__.py", 
    f"{project_name}/utils/common.py",
    
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",
     
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/config_entity.py",
     
    f"{project_name}/constants/__init__.py",
    
    f"{project_name}/cloud/__init__.py",
    "Scoring_Data/",
    "notebooks/",
    "templates/",
    "app.py",
    "Dockerfile",
    # "requierements.txt",
    # "setup.py",
]

# Loop to create all the dirs and files above
for item in list_of_files:
    path = Path(item)
    if item.endswith("/"):  # es un directorio
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created: {path}")
    else:  # es un archivo
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            path.touch()
            logging.info(f"File created: {path}")
        else:
            logging.info(f"File already exists: {path}")