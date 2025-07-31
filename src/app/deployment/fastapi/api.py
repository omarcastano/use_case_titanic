"""
FastAPI application for Titanic survival prediction using Pydantic models.
This module defines a FastAPI application with endpoints for making predictions
using a trained Titanic survival model. It uses Pydantic for data validation
and serialization.
"""

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.app.model.model import TitanicModel

# load trained model
model = TitanicModel()
model.load_model("src/app/model/trained_models/titanic_model.joblib")

app = FastAPI()


class InputData(BaseModel):
    sex: str = "male"
    pclass: int = 3
    age: float = 30.0


class OutputData(BaseModel):
    prediction: int
    probability: float


@app.post("/post_predict")
def post_predict(data: InputData) -> OutputData:
    """Endpoint to make predictions using the Titanic survival model.
    Arguments:
    ----------
    data : InputData
        The input data containing features for prediction.
    Returns:
    -------
    dict
        A dictionary containing the prediction result.

    Example Usage:
    --------------
    To make a POST request with JSON data, you can use the following curl command:
        curl -X 'POST' \
        'http://127.0.0.1:8000/post_predict' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "sex": "male",
        "pclass": 3,
        "age": 30
        }'
    """
    # Convert input data to DataFrame
    data_dict = data.model_dump()
    data = pd.DataFrame([data_dict])

    prediction = int(model.predict(data)[0])
    probability = round(model.predict_proba(data)[0], 2)

    return OutputData(prediction=prediction, probability=probability)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", port=8000, host="127.0.0.1", reload=True)
