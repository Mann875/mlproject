import os
import dill
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # getting the directory path from the file path, so that we can create the directory if it does not exist.
        os.makedirs(dir_path, exist_ok=True) # creating the directory if it does not exist, and if it already exists, it will keep it as it is and not raise an error.
        with open(file_path, 'wb') as file_obj: # opening the file in write binary mode, so that we can save the object in binary format.
            dill.dump(obj, file_obj) # saving the object to the file using dill module, which is used for serializing and deserializing Python objects.
    except Exception as e:
        raise CustomException(e, sys) # raising a custom exception if any error occurs during the saving of the object, and passing the error message and the system information to the custom exception class.

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {} # initializing an empty dictionary to store the model report, which will contain the model name and the corresponding r2 score.
        for i in range(len(models)):
            model = list(models.values())[i] # getting the model from the models dictionary using the index, and storing it in the model variable.
            
            # Just as we listed all the model, List all the params
            para = params[list(models.keys())[i]] # getting the parameters for the model from the params dictionary using the model name as the key, and storing it in the para variable.   
            
            gs = GridSearchCV(model, para, cv=3) # creating an instance of the GridSearchCV class, and passing the model, parameters and the number of cross validation folds as arguments to the class, so that it can perform hyperparameter tuning on the model using grid search cross validation.
            gs.fit(X_train, y_train) # fitting the GridSearchCV object on the training data, so that it can find the best parameters for the model based on the training data.
        
            model.set_params(**gs.best_params_) # setting the best parameters found by the GridSearchCV object to the model, so that we can use the best parameters for training the model on the training data.    
            model.fit(X_train, y_train) # fitting the model on the training data, so that we can make predictions on the test data.
           
            y_train_pred = model.predict(X_train) # making predictions on the training data using the fitted model, and storing the predicted values in the y_train_pred variable.
            train_model_score = r2_score(y_train, y_train_pred) # calculating the r2 score for the training data by comparing the actual values (y_train) with the predicted values (y_train_pred), and storing it in the train_model_score variable.
        
            y_test_pred = model.predict(X_test) # making predictions on the test data using the fitted model, and storing the predicted values in the y_test_pred variable.
            test_model_score = r2_score(y_test, y_test_pred) # calculating the r2 score for the test data by comparing the actual values (y_test) with the predicted values (y_test_pred), and storing it in the test_model_score variable.
            
            model_report[list(models.keys())[i]] = test_model_score # adding the model name and its corresponding r2 score to the model_report dictionary, where the key is the model name and the value is the r2 score.
        return model_report # returning the model_report dictionary, which contains all the models and their corresponding r2 scores.
    except Exception as e:
        raise CustomException(e, sys) # raising a custom exception if any error occurs during the evaluation of the models, and passing the error message and the system information to the custom exception class. 
    