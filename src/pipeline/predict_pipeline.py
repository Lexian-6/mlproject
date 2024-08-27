import os
import pandas as pd

from src.exception import CustomizedException
from src.utils import load_object
from src.logger import logging

from dataclasses import dataclass


class PredictPipelineConfig:
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

class PredictPipeline:
    def __init__(self) -> None:
        self.predict_pipeline_config = PredictPipelineConfig()
    def predict(self, raw_features):
        try:
            model = load_object(self.predict_pipeline_config.model_path)
            preprocessor = load_object(self.predict_pipeline_config.preprocessor_path)
            logging.info("Load model and preprocessor.")
            processed_features = preprocessor.transform(raw_features)
            preds = model.predict(processed_features)
            logging.info("Make prediction.")
            return preds
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            raise CustomizedException(e)
        
class CustomData:
    def __init__(self,
                 gender:str, 
                 race_ethnicity:str, 
                 parental_level_of_education:str, 
                 lunch:str, 
                 test_preparation_course:str, 
                 reading_score:int, 
                 writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    def get_data_as_data_frame(self):
        try:
            df=pd.DataFrame({
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            })
            return df
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            raise CustomizedException(e)