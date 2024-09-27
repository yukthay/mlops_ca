from churn_predictor import ChurnPredictor

class MonthToMonth(ChurnPredictor):
    def __init__(self):
        super().__init__('Month-to-Month')

    def additional_feature(self):
        return "This is a Month-to-Month contract."
