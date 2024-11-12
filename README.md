# Student Performance Prediction

This project aims to predict student performance based on various input factors using machine learning models. The application predicts the student's math score using data about their demographics, parental education, test preparation courses, and exam scores.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Project Overview

The Student Performance Prediction project is built using Flask for the web interface and various machine learning models for prediction. The application takes input parameters such as gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score, and writing score to predict the math score.

## Features

- Predict student math scores based on demographic and performance data.
- User-friendly web interface to input data and view predictions.
- Multiple machine learning models implemented for accurate prediction.
- Data preprocessing pipeline for scaling and transforming input data.

## Technologies Used

- **Python**: Core language used for backend and model development.
- **Flask**: Used to create the web application.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-Learn**: For model training, evaluation, and preprocessing.
- **CatBoost, XGBoost**: Advanced machine learning models used for regression.
- **HTML/CSS**: For designing the web interface.

## Project Structure
  ```
  . ├── src
    │ ├── pipeline 
    │ │   ├── predict_pipeline.py # Prediction pipeline and data handling classes 
    │ │   ├── train_pipeline.py # Model training pipeline 
    │ ├── components 
    │ │   ├── data_ingestion.py # Handles data loading 
    │ │   ├── data_preprocessing.py # Data preprocessing logic 
    │ │   ├── model_trainer.py # Model training and evaluation 
    │ ├── utils.py # Utility functions (load/save models, etc.) 
    │ ├── exception.py # Custom exception handling 
    │ ├── logger.py # Logging configurations 
    ├── templates 
    │     ├── index.html # Home page of the application 
    │     ├── home.html # Form to input student data for prediction 
    ├── artifacts 
    │    ├── model.pkl # Trained model file 
    │    ├── preprocessor.pkl # Preprocessing object 
    ├── app.py # Main application file 
    ├── README.md # Project documentation 
    ├── requirements.txt # List of dependencies
  ```

## Setup Instructions

To set up the project, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/student-performance-prediction.git
   cd student-performance-prediction

2. **Create a Virtual Environment (optional but recommended):**

    ```
    python -m venv venv 
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
3. **Install Required Packages: You can install the necessary packages using pip:**

   ``` 
   pip install -r requirements.txt

**The requirements.txt should include:**

    
    pandas
    scikit-learn
    catboost
    xgboost
    Flask

## Usage

### Training the Model

To train the model, run the training pipeline:

    python src/train_pipeline.py

This will:

* Load the dataset from data/student_data.csv.
* Split the data into training and testing sets (80% for training and 20% for testing).
* Train a RandomForestRegressor model on the training data.
* Evaluate the model's performance using the R² score on the test data.
* Save the trained model to artifacts/model.pkl for later use.

### Making Predictions

To make predictions using the trained model, run the Flask web application: 

    python app.py

## How It Works:

* Data Loading: The project loads data from a CSV file containing student performance metrics.
* Data Preprocessing: The data is preprocessed to handle missing values and categorical variables.
* Model Training: A machine learning model (e.g., Random Forest) is trained using the preprocessed data.
* Model Evaluation: The trained model is evaluated using metrics such as R² score.
* Prediction Interface: A Flask web application allows users to input data and receive predictions.

## Future Enhancements

* Model Tuning: Implement hyperparameter tuning to improve model accuracy.
* Additional Features: Explore more features that could influence student performance (e.g., study time, school resources).
* Deployment: Deploy the model to a cloud platform for broader accessibility.
* User Authentication: Implement user authentication for personalized predictions and history tracking.

## Contributing

Contributions are welcome! If you’d like to contribute to this project, please follow these steps:

* Fork the repository.
* Create a new branch (e.g., feature/YourFeature).
* Make your changes.
* Commit your changes and push to your branch.
* Create a pull request.

Please ensure your code adheres to the project's coding standards and is well-documented.

## Acknowledgments

* Scikit-learn
* Pandas
* Flask
* CatBoost
* XGBoost
