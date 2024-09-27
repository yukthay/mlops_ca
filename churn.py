import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ChurnPredictor:
    def __init__(self, customer_data):
        """
        Initialize with a DataFrame that contains customer data.
        Columns: ['Customer_ID', 'Contract_Type', 'Monthly_Charges', 'Tenure', 'Churn_Flag']
        """
        self.customer_data = customer_data
        self.model = None

    def preprocess_data(self):
        """
        Preprocess the data: convert Contract_Type to dummy variables and
        separate features from the target (Churn_Flag).
        """
        df = self.customer_data.copy()
        # One-hot encode Contract_Type
        df = pd.get_dummies(df, columns=['Contract_Type'], drop_first=True)

        # Features and target
        X = df.drop(columns=['Customer_ID', 'Churn_Flag'])
        y = df['Churn_Flag']

        return X, y

    def train_model(self):
        """
        Train a simple Logistic Regression model to predict churn.
        """
        X, y = self.preprocess_data()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the logistic regression model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")

    def predict_churn(self, new_customer_data):
        """
        Predict churn probability for a new customer based on their data.
        """
        if not self.model:
            raise ValueError("Model is not trained yet. Please train the model before predicting.")

        # Convert the new customer data into a DataFrame
        new_customer_df = pd.DataFrame([new_customer_data])
        new_customer_df = pd.get_dummies(new_customer_df, columns=['Contract_Type'], drop_first=True)

        # Align columns with the training data
        X, _ = self.preprocess_data()
        new_customer_df = new_customer_df.reindex(columns=X.columns, fill_value=0)

        # Predict churn probability
        churn_probability = self.model.predict_proba(new_customer_df)[:, 1][0]
        return churn_probability

    def calculate_retention_rate(self):
        """
        Calculate and return the retention rate based on the Churn_Flag in the dataset.
        Retention rate = (Number of non-churned customers / Total customers) * 100
        """
        total_customers = len(self.customer_data)
        non_churned_customers = len(self.customer_data[self.customer_data['Churn_Flag'] == 0])

        retention_rate = (non_churned_customers / total_customers) * 100
        return retention_rate


# Example usage:
if __name__ == "__main__":
    # Sample customer data
    data = pd.read_csv('C:/Users/yukth/OneDrive/Desktop/mlops_ca/customer_data.csv')

    customer_df = pd.DataFrame(data)
    churn_predictor = ChurnPredictor(customer_df)
    
    # Train the model
    churn_predictor.train_model()

    # Predict churn for a new customer
    new_customer = {'Customer_ID': 6, 'Contract_Type': 'Month-to-Month', 'Monthly_Charges': 50.00, 'Tenure': 12}
    probability = churn_predictor.predict_churn(new_customer)
    print(f"Churn probability for new customer: {probability:.2f}")

    # Calculate retention rate
    retention_rate = churn_predictor.calculate_retention_rate()
    print(f"Customer retention rate: {retention_rate:.2f}%")
