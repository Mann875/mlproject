import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
import sklearn
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,  
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):  # The train and test arrays are being returned from the data transformation file.

        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], #Take out the last column as the target variable and the rest of the columns as the features., and feed everything in to train array.
                train_array[:, -1], # and now from train array take every rows. and take the last column as the target variable.
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # To get best model score from the dictionary.
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from the dictionary.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)  # Based on the index of best model score, we will get the best model name.
                ]
            best_model = models[best_model_name] # Based on the best model name, we will get the best model from the models dictionary. 

            if best_model_score < 0.6: # If the best model score is less than 0.6, then we will raise an exception.
                raise CustomException("No best model found with score greater than 0.6", sys)   
            

            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, # Saving the best model to the file path specified in the model trainer config class.
                obj=best_model, # Saving the best model object to the file. 
            )

            predicted = best_model.predict(X_test) # Making predictions on the test data using the best model, and storing the predicted values in the predicted variable.
            r2_square = r2_score(y_test, predicted) # Calculating the r2 score for the test data by comparing the actual values (y_test) with the predicted values (predicted), and storing it in the r2_square variable.
            return r2_square # Returning the r2 score for the test data.
        

        except Exception as e:
            raise CustomException(e, sys)
