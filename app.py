from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__) # To run the app.py file, we need to set the FLASK_APP environment variable to app.py and then run the flask command. 
# This is the entry point of the application, and it will run the app.py file when we run the flask command.

app = application
@app.route('/') # This is the route for the home page of the application, and it will render the index.html file when we access the home page.
def index():
    return render_template('index.html') # This will render the index.html file when we access the home page.
    
@app.route('/predictdata', methods=['GET','POST']) # This is the route for the predict page of the application, and it will render the predict.html file when we access the predict page.
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') # This will render the home.html file when we access the predict page.    
    else: 
        data=CustomData(
            # We will read all the data from the form in the home.html file, and we will pass it to the CustomData class, which will convert it into a dataframe format, and then we will pass that dataframe to the PredictPipeline class, which will make predictions using the trained model and return the predicted value. 
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe() # This will convert the input data into a dataframe format, which is required for making predictions using the model.
        print(pred_df)
        predict_pipeline = PredictPipeline() # This will create an instance of the PredictPipeline class, which will be used to make predictions using the trained model.
        results = predict_pipeline.predict(pred_df) # This will make predictions using the trained model and the input data, and it will return the predicted value.
        return render_template('home.html', results=results[0]) # This will render the home.html file and pass the predicted value to the home.html file, so that we can display the predicted value on the web page.   

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True) # This will run the Flask application in debug mode, which will allow us to see the error messages in the console if any error occurs during the execution of the application.      
