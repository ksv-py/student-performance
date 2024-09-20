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
from utils import evaluate_model,save_object

@dataclass
class ModelTrainerConfig:
    model_trained_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Initiating Model Training")
            X_train, y_train , X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Split training and test input data")


            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "Catboost": CatBoostRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            logging.info("Cretaing model accuuracy report")
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            logging.info(f"Best Model Selected: {best_model_name}:{best_model_score}")
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best found model in both training and test datasets")

            save_object(self.modeltrainerconfig.model_trained_path, best_model)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)