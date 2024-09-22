from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    """
    Renders the index page of the application.
    """
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['GET', 'POST']) 
def predict_datapoint():
    """
    Handles the prediction request. Renders the prediction form on GET request
    and processes the prediction on POST request.
    """
    if request.method == 'GET':
        # Render the home page form
        return render_template('home.html')
    else:
        # Create an instance of CustomData with the input values from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),  # Convert score inputs to integers
            writing_score=int(request.form.get('writing_score'))
        )

        # Convert the custom data into a DataFrame
        pred_df = data.get_data_as_dataframe()
        print(pred_df)  # Debugging print

        # Instantiate PredictPipeline and get predictions
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Render the home page with prediction results
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0", debug=True)
