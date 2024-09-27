from churn_predictor import ChurnPredictor

class TwoYear(ChurnPredictor):
    def __init__(self):
        super().__init__('Two-Year')

    def additional_feature(self):
        return "This is a Two-Year contract."
