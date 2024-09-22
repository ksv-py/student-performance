import sys
import pandas as pd
from pathlib import Path

# Adding the parent directory to the system path to access other modules in the project
sys.path.append(str(Path(__file__).parent.parent))

# Importing custom exception handling and logging modules
from exception import CustomException
from logger import logging
from utils import load_object

class PredictPipeline:
    """
    PredictPipeline class is responsible for handling the prediction process.
    It loads the pre-trained model and preprocessor, applies the preprocessor 
    to the input features, and returns the predictions.
    """
    def __init__(self):
        pass
    
    def predict(self, features):
        """
        Predicts the output using the trained model and preprocessor.

        Args:
            features (pd.DataFrame): DataFrame containing the features for prediction.

        Returns:
            np.array: Array of predictions.

        Raises:
            CustomException: If any error occurs during the prediction process.
        """
        try:
            # Define paths to the model and preprocessor objects
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            # Load the trained model and preprocessor objects
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict the target variable using the loaded model
            preds = model.predict(data_scaled)

            if preds[0] > 100:
                preds[0] = 100  # Cap the prediction at 100 if it goes beyond
            elif preds[0] < 0:
                preds[0] = 0 
                
            return preds
        
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

class CustomData:
    """
    CustomData class is responsible for capturing the input data from the user
    and converting it into a format suitable for prediction, specifically a DataFrame.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        """
        Initializes the CustomData object with the required input fields.

        Args:
            gender (str): Gender of the student.
            race_ethnicity (str): Race/ethnicity group of the student.
            parental_level_of_education (str): Education level of the student's parents.
            lunch (str): Type of lunch the student receives (e.g., standard, free/reduced).
            test_preparation_course (str): Whether the student completed a test preparation course.
            reading_score (int): Reading score of the student.
            writing_score (int): Writing score of the student.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Converts the captured input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the input data.

        Raises:
            CustomException: If an error occurs during DataFrame creation.
        """
        try:
            # Creating a dictionary to convert input data to DataFrame format
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            # Returning the input data as a DataFrame
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            # Raise a custom exception if an error occurs during the DataFrame creation
            raise CustomException(e, sys)
