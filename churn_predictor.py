class ChurnPredictor:
    def __init__(self, contract_type):
        self.contract_type = contract_type

    def predict(self, features):
        # Dummy implementation of prediction logic
        if self.contract_type == 'Month-to-Month':
            return "High Churn Risk"
        elif self.contract_type == 'One-Year':
            return "Medium Churn Risk"
        elif self.contract_type == 'Two-Year':
            return "Low Churn Risk"
        return "Unknown Contract Type"
