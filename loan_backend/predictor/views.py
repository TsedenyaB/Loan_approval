from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd

from .ml_loader import lr_model, dt_model


@api_view(["POST"])
def predict_loan(request):
    data = request.data

    df = pd.DataFrame([data])

    lr_pred = lr_model.predict(df)[0]
    dt_pred = dt_model.predict(df)[0]

    return Response({
        "logistic_regression": "Approved" if lr_pred == 1 else "Rejected",
        "decision_tree": "Approved" if dt_pred == 1 else "Rejected"
    })
