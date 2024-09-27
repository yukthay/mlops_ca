import unittest
import pandas as pd
from churn import ChurnPredictor  # Adjust the import to your actual file structure

class TestChurnPredictor(unittest.TestCase):
    def setUp(self):
        # Sample customer data for testing
        self.data = pd.DataFrame({
            'Customer_ID': [1, 2, 3, 4, 5],
            'Contract_Type': ['Month-to-Month', 'One Year', 'Two Year', 'Month-to-Month', 'One Year'],
            'Monthly_Charges': [50.00, 60.00, 70.00, 45.00, 80.00],
            'Tenure': [12, 24, 36, 12, 48],
            'Churn_Flag': [0, 1, 0, 0, 1]  # 3 retained (0) and 2 churned (1)
        })
        self.predictor = ChurnPredictor(self.data)
        self.predictor.train_model()  # Ensure the model is trained

    def test_predict_churn(self):
        # Test predicting churn for an existing customer
        customer = self.data.iloc[0]  # Using the first customer's data
        result = self.predictor.predict_churn(customer)
        self.assertIsInstance(result, float, "Prediction should be a float value representing probability.")

    def test_calculate_retention_rate(self):
        # Calculate expected retention rate: (3 non-churned / 5 total) * 100
        expected_retention_rate = (3 / 5) * 100  # 3 retained out of 5
        calculated_retention_rate = self.predictor.calculate_retention_rate()
        self.assertAlmostEqual(calculated_retention_rate, expected_retention_rate, places=2)

    def test_no_customers(self):
        # Test handling of empty dataset
        empty_data = pd.DataFrame(columns=['Customer_ID', 'Contract_Type', 'Monthly_Charges', 'Tenure', 'Churn_Flag'])
        empty_predictor = ChurnPredictor(empty_data)
        with self.assertRaises(ZeroDivisionError):
            empty_predictor.calculate_retention_rate()  # This should raise an error due to division by zero

if __name__ == '__main__':
    unittest.main()
