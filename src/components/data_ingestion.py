import os
import sys
from src.exception import CustomException # importing custom exception class
from src.logger import logging # importing logging module
import pandas as pd # Because we are going to work with dataframes
from sklearn.model_selection import train_test_split # for splitting the data into train and test sets
from dataclasses import dataclass # for creating class variables

from src.components.data_transformation import DataTransformation, DataTransformationConfig # importing the data transformation config class, because we need to use the preprocessor object file path from the data transformation config class to save the preprocessor object in the same path as we are saving the transformed data, so that we can easily load the preprocessor object and the transformed data in the later stages of the pipeline.  


@dataclass   # Decorator to automatically generate special methods like __init__() and __repr__() for the class
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv') # path to save the train data
    test_data_path: str = os.path.join('artifacts', 'test.csv') # path to save the test data
    raw_data_path: str = os.path.join('artifacts', 'data.csv') # path to save the raw data

# So this are the inputs given to the data ingestion component, so that it knows where to save the train, test and raw data.

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # creating an instance of the DataIngestionConfig class

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component") # logging the start of the data ingestion process
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe") # logging the successful reading of the dataset, to maintain logs
            # Only minor change in the above two line will help the user to input data from any sources like local file, database or cloud storage, and we can also add some data validation steps here to check the quality of the data before saving it to the specified paths.   
            # We can also add some data transformation steps here to preprocess the data before saving it to the specified paths, like handling missing values, encoding categorical variables, feature scaling, etc.   
                 

            # NOw to store data we need to create the directory, if it does not exist = artifact
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # creating the directory if it does not exist    
            # exist_ok=True means that if the directory already exists, it will keep it as it is and not raise an error.

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # saving the raw data to the specified path
            logging.info("train_test_split initiated") # logging the successful saving of the raw data   
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Train_test split done and saving the train and test data to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # saving the train data to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # saving the test data to the specified path 
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) # returning the paths of the train and test data
        
        except Exception as e:
            raise CustomException(e,sys) # raising a custom exception if any error occurs during the data ingestion process, and passing the error message and the system information to the custom exception class 
        

if __name__ == "__main__":
    obj = DataIngestion() # creating an instance of the DataIngestion class
    train_data, test_data = obj.initiate_data_ingestion() # calling the initiate_data_ingestion method to start the data ingestion process  
    data_transformation = DataTransformation() # creating an instance of the DataTransformation class
    data_transformation.initiate_data_transformation(train_data, test_data) 
    # calling the initiate_data_transformation method to start the data transformation process, and passing the paths of the train and test data as arguments to the method, so that it can read the train and test data from the specified paths and perform the data transformation steps on them.
    

# So the data ingestion step will be executed when we run this script, and artifact folder will be created after this step
