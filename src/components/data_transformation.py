import os
import sys
import pickle
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_path: str = os.path.join("artifacts", "train_transformed.npy")
    transformed_test_path: str = os.path.join("artifacts", "test_transformed.npy")


class DataTransformation:
    '''This class is responsible for performing data transformation on the train and test data, and saving the preprocessor object and the transformed data to the specified paths. The data transformation steps include handling missing values, encoding categorical variables, and feature scaling.'''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                    # We created two steps in the pipleline 1. is handling missing values and 2. is standard scaling.
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')), # This will handle the unknown categories in the test data that are not present in the train data, and it will ignore them instead of raising an error.
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # As we created all the needed pipelines for numerical and categorical columns, now we will combine them using ColumnTransformer, so that we can apply the transformations to the respective columns in one go. 
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns), # Pipeline name, what our pipeline is, and which columns to apply.
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    '''Start data transformation process, by calling the get_data_transformer_object method to get the preprocessor object, and then applying the transformations to the train and test data, and saving the transformed data and the preprocessor object to the specified paths.   '''

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name] # separating the input features and target feature from the train data

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name] # separating the input features and target feature from the test data.


            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) # applying the transformations to the input features of the train data
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) # applying the transformations to the input features of the test data.

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # combining the transformed input features and target feature of the train data.
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # combining the transformed input features and target feature of the test data.  
            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path  
            )

        except Exception as e:
            raise CustomException(e,sys)    