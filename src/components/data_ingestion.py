import os
import sys
import pandas as pd 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add the parent directory to the system path to allow imports from local modules.
sys.path.append(str(Path(__file__).parent.parent))

# Import necessary custom classes and modules for exception handling, logging, data transformation, and model training.
from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.

    Attributes:
        train_data_path (str): Path to save the training data.
        test_data_path (str): Path to save the testing data.
        raw_data_path (str): Path to save the raw data.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """
    Class for handling the ingestion of data from a CSV file.

    This class reads the dataset, splits it into training and testing sets, 
    and saves these sets to specified paths.

    Methods:
        initiate_data_ingestion: Reads the dataset, splits it, and saves the results.
    """
    def __init__(self):
        """Initializes the DataIngestion class and its configuration."""
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """
        Reads the dataset, performs train-test split, and saves the resulting datasets.

        Returns:
            tuple: Paths of the training and testing datasets.

        Raises:
            CustomException: If an error occurs during data ingestion.
        """
        logging.info("Initiated Data Ingestion")
        try:
            # Read the dataset from a specified CSV file.
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset")

            # Create the directory for saving the raw dataset if it doesn't exist.
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # Save the raw dataset to the specified path.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Created Raw Dataset')

            # Split the dataset into training and testing sets (80-20 split).
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Initiated Train Test Split')

            # Create the directory for saving the training dataset if it doesn't exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Save the training dataset to the specified path.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info('Created Train Dataset')

            # Create the directory for saving the testing dataset if it doesn't exist.
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            # Save the testing dataset to the specified path.
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Created Test Dataset')

            logging.info("Successfully completed Data Ingestion")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Log the error and raise a custom exception if an error occurs.
            logging.error("ERROR in executing the script")
            raise CustomException(e, sys)
        
# Main execution block
if __name__ == "__main__":
    # Create an instance of DataIngestion and initiate data ingestion.
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of DataTransformation to transform the data.
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

    # Create an instance of ModelTrainer and initiate model training.
    modeltrainer = ModelTrainer()
    print(f'Model Accuracy: {modeltrainer.initiate_model_training(train_arr, test_arr)}')
