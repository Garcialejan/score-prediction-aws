import os
import sys
import json
import pandas as pd
import pymongo
from src.exception.exception import ScorePredictionException
from src.constants import DATA_FILE_PATH, DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

from dotenv import load_dotenv
load_dotenv()
username = os.getenv("db_username")
password = os.getenv("db_password")
if not username or not password:
    raise ScorePredictionException("Database credentials are missing.", sys)
mongodb_uri = f"mongodb+srv://{username}:{password}@scoringproject.lcjwpi4.mongodb.net/?appName=ScoringProject"

import certifi
ca = certifi.where() 

class ScoringDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ScorePredictionException(e, sys)
        
    @staticmethod
    def csv_to_json(data_path):
        try:
            df = pd.read_csv(data_path, sep = ",", index_col = False)
            records = list(json.loads(df.T.to_json()).values())
            return records
        except FileNotFoundError:
            raise ScorePredictionException("File not found.", sys)
        except Exception as e:
            raise ScorePredictionException(f"Error processing CSV: {e}", sys)

    def insert_data_mongodb(self, records, database, collection):
        """
        Take the data into the database

        Args:
            records (str): data
            database (str): databse name
            collection (str): is like the table name in SQL tables
        """
        try:
            self.records = records
            self.database = database
            self.collection = collection
            
            self.mongo_client = pymongo.MongoClient(mongodb_uri, tlsCAFile=ca)  # Usa certificados seguros
            self.database = self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        
        except pymongo.errors.ConnectionFailure:
            raise ScorePredictionException("Failed to connect to MongoDB.", sys)
        except Exception as e:
            raise ScorePredictionException(f"Error inserting data into MongoDB: {e}", sys)
        
        
if __name__ =="__main__":    
    scoringobj = ScoringDataExtract()
    records = scoringobj.csv_to_json(DATA_FILE_PATH)
    print(records)
    no_of_records = scoringobj.insert_data_mongodb(records,
                                                   database = DATA_INGESTION_DATABASE_NAME,
                                                   collection = DATA_INGESTION_COLLECTION_NAME)
    print(no_of_records)