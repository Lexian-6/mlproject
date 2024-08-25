import os

import dill

from src.exception import CustomizedException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.error(f"An error occurred: {CustomizedException(e)}")
        raise CustomizedException(e)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error(f"An error occurred: {CustomizedException(e)}")
        raise CustomizedException(e)