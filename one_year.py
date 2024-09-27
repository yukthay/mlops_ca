from churn_predictor import ChurnPredictor

class OneYear(ChurnPredictor):
    def __init__(self):
        super().__init__('One-Year')

    def additional_feature(self):
        return "This is a One-Year contract."