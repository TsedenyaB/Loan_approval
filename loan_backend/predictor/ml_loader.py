import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

lr_model = joblib.load(
    os.path.join(BASE_DIR, "predictor/ml_models/logistic_regression_model.pkl")
)

dt_model = joblib.load(
    os.path.join(BASE_DIR, "predictor/ml_models/decision_tree_model.pkl")
)
