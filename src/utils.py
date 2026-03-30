import os
import dill
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # getting the directory path from the file path, so that we can create the directory if it does not exist.
        os.makedirs(dir_path, exist_ok=True) # creating the directory if it does not exist, and if it already exists, it will keep it as it is and not raise an error.
        with open(file_path, 'wb') as file_obj: # opening the file in write binary mode, so that we can save the object in binary format.
            dill.dump(obj, file_obj) # saving the object to the file using dill module, which is used for serializing and deserializing Python objects.
    except Exception as e:
        raise CustomException(e, sys) # raising a custom exception if any error occurs during the saving of the object, and passing the error message and the system information to the custom exception class.

