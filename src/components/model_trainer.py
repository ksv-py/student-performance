import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the ModelTrainer.

    Attributes:
        model_trained_path (str): Path to save the trained model object.
    """
    model_trained_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    A class used to train various regression models, perform hyperparameter tuning,
    evaluate model performance, and save the best model.

    Methods:
        initiate_model_training(train_array, test_array): Trains models, performs hyperparameter tuning,
        and returns the R2 score of the best model.
    """

    def __init__(self):
        """
        Initializes the ModelTrainer with the model training configuration.
        """
        self.modeltrainerconfig = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Initiates the training and evaluation of multiple regression models with hyperparameter tuning.

        Args:
            train_array (np.ndarray): Training data array containing input features and target variable.
            test_array (np.ndarray): Testing data array containing input features and target variable.

        Returns:
            float: R2 score of the best-performing model on the test data.

        Raises:
            CustomException: If no suitable model is found or any exception occurs during model training.
        """
        try:
            logging.info("Initiating Model Training")

            # Splitting the training and test data into input features and target variable
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Input features for training
                train_array[:, -1],   # Target variable for training
                test_array[:, :-1],   # Input features for testing
                test_array[:, -1]     # Target variable for testing
            )
            logging.info("Split training and test input data")

            # Dictionary of models to be trained
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            logging.info("Setting up Parameters for Hyperparameter tuning")

            # Hyperparameters for tuning each model
            params = {
                "Decision Tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
                "K-Neighbors": {},        # Default hyperparameters for K-Neighbors
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate the models with the provided hyperparameters
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            logging.info("Creating model accuracy report")

            # Find the best model based on the highest score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best Model Selected: {best_model_name}:{best_model_score}")

            # Raise an exception if no suitable model is found
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info("Best found model in both training and test datasets")

            # Save the best model object to a file
            save_object(self.modeltrainerconfig.model_trained_path, best_model)

            # Predict the test data using the best model and calculate the R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # Raise a custom exception if any error occurs during the process
            raise CustomException(e, sys)
