import sys # for handling exceptions and getting system information.
import pandas as pd # for data manipulation and analysis.
from src.exception import CustomException # for handling custom exceptions.
from src.logger import logging # for logging information and errors.
from src.utils import load_object # for loading the saved model and preprocessor objects.


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features): # This function will take the input features from the user, and it will return the predicted value using the trained model.
        try:
            model_path = 'artifacts/model.pkl' # The path where the trained model is saved.
            preprocessor_path = 'artifacts/preprocessor.pkl' # The path where the preprocessor object is saved.
            model = load_object(file_path=model_path) # Load the trained model using the load_object function from utils.py
            preprocessor = load_object(file_path=preprocessor_path) # Load the preprocessor object using the load_object function from utils.py
            data_scaled = preprocessor.transform(features) # Scale the input features using the preprocessor object.
            preds = model.predict(data_scaled) # Make predictions using the trained model and the scaled input features.
            return preds
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:   # It will be responsible for mapping all the inputs given to the html to the backend
    def __init__(self,
        # DEfine all the input features that we are going to take from the user in the html file.
        gender:str,
        race_ethnicity:str,
        parental_level_of_education,    
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int):

        # THis all values are coming from web application.
        # Initialize all the input features as instance variables.
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch      
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_dataframe(self): # This function will convert the input data into a dataframe format, which is required for making predictions using the model.
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)