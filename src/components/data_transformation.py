import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path

# Adding the parent directory to the system path for importing custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Importing custom exception handling, logging, and utility functions
from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.

    Attributes:
        preprocessor_obj_file_path (str): Path to save the preprocessor object file.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Class for handling data transformation processes, including preprocessing
    numerical and categorical features using pipelines.

    Methods:
        get_data_transformer_object: Creates and returns a preprocessor object.
        initiate_data_transformation: Transforms train and test data using the preprocessor.
    """
    def __init__(self):
        """Initializes the DataTransformation class with its configuration."""
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and configures a preprocessing object for numerical and categorical features.

        This function creates two pipelines:
        - Numerical pipeline: Imputes missing values with the median and scales the data.
        - Categorical pipeline: Imputes missing values with the most frequent value, 
          encodes the categories, and scales the encoded values.

        Returns:
            ColumnTransformer: A preprocessor object with configured pipelines for transformation.

        Raises:
            CustomException: If any error occurs during the creation of the preprocessor object.
        """
        try:
            # Define numerical and categorical features to be preprocessed
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                    'lunch', 'test_preparation_course']
            
            # Pipeline for numerical features: impute missing values and scale them
            num_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='median')),
                    ('Standard Scaler', StandardScaler())
                ]
            )

            # Pipeline for categorical features: impute missing values, encode, and scale them
            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('One Hot Encoder', OneHotEncoder(sparse_output=True)),
                    ('Standard Scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            # Combine the numerical and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            # Raise a custom exception if an error occurs during the process
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads the training and testing datasets, applies transformations, and saves the preprocessor object.

        Args:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the testing data CSV file.

        Returns:
            tuple: Transformed training and testing data arrays.

        Raises:
            CustomException: If any error occurs during the data transformation process.
        """
        try:
            logging.info("Initiating Data Transformation")
            # Read the training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data completed")
            logging.info("Obtaining preprocessing objects")

            # Get the preprocessor object configured with pipelines
            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math_score'  # Define the target column

            # Split input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            # Split input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # Fit the preprocessor on training data and transform both training and testing data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed input features with target values for train and test sets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save the preprocessor object to the specified path
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                        obj=preprocessing_obj)

            return train_arr, test_arr

        except Exception as e:
            # Raise a custom exception if an error occurs during the transformation process
            raise CustomException(e, sys)
