import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Class to store and save the model details along with evaluation metrics
class ModelObject:
    def __init__(self, model_name, model, params, best_params, evaluation_metrics, version):
        self.model_name = model_name
        self.model = model
        self.params = params
        self.best_params = best_params
        self.evaluation_metrics = evaluation_metrics
        self.version = version

    def log_details(self):
        log_message = f"Model: {self.model_name} (Version: {self.version})\n"
        log_message += f"Initial Parameters: {self.params}\n"
        log_message += f"Best Parameters after tuning: {self.best_params}\n"
        log_message += f"Evaluation Metrics: {self.evaluation_metrics}\n"
        return log_message

    def save(self, save_path):
        joblib.dump(self, save_path)
        print(f"Model saved at: {save_path}")

# Base Class for Dataset Handling
class Dataset:
    def __init__(self):
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        # Load your customer dataset (replace this with actual data loading)
        self.data = pd.read_csv('customer_data.csv')  # Ensure the correct path to your CSV file
        self.target = 'Churn_Flag'  # Define your target variable

    def preprocess(self):
        # Convert categorical variables to dummy variables
        self.data = pd.get_dummies(self.data, columns=['Contract_Type'], drop_first=True)

        # Train-test split
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

# Base Class for Model Selection and Tuning
class ModelSelector:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC(),
            'LogisticRegression': LogisticRegression(max_iter=200)
        }
        self.best_model_object = None
        self.version = 1  # Versioning starts at 1

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def select_model(self, X_train, y_train, X_test, y_test):
        # Define parameter grids for each model
        param_grids = {
            'RandomForest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'LogisticRegression': {'C': [0.01, 0.1, 1]}
        }

        best_score = 0
        for model_name, model in self.models.items():
            print(f"Tuning {model_name}...")
            tuned_model, best_params = self.hyperparameter_tuning(model, param_grids[model_name], X_train, y_train)
            
            # Evaluate on test data
            y_pred = tuned_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            evaluation_metrics = classification_report(y_test, y_pred, output_dict=True)

            print(f"{model_name} Test Accuracy: {accuracy}")

            # Save model object only if it is the best one
            if accuracy > best_score:
                best_score = accuracy
                self.best_model_object = ModelObject(
                    model_name=model_name,
                    model=tuned_model,
                    params=param_grids[model_name],
                    best_params=best_params,
                    evaluation_metrics={"accuracy": accuracy, "classification_report": evaluation_metrics},
                    version=self.version
                )

        print(f"Best Model: {self.best_model_object.model_name}")
        return self.best_model_object

    def save_best_model(self):
        if self.best_model_object:
            # Create the model's versioned file name
            save_path = f"models/{self.best_model_object.model_name}_v{self.version}.pkl"
            self.best_model_object.save(save_path)
            self.version += 1  # Increment the version for the next save


# Main AutoML Pipeline
class AutoMLPipeline:
    def __init__(self):
        self.dataset = Dataset()
        self.model_selector = ModelSelector()

    def run(self):
        # Load and preprocess data
        print("Loading and Preprocessing Data...")
        self.dataset.load_data()
        self.dataset.preprocess()

        # Model Selection and Evaluation
        print("Selecting the best model...")
        best_model = self.model_selector.select_model(
            self.dataset.X_train, self.dataset.y_train, 
            self.dataset.X_test, self.dataset.y_test
        )

        # Save the best model with versioning
        self.model_selector.save_best_model()


# Run the AutoML pipeline
if __name__ == "__main__":
    pipeline = AutoMLPipeline()
    pipeline.run()
