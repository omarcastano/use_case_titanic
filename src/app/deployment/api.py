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


@app.post("/post_predict")
def post_predict(data: InputData) -> dict:
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

    prediction = model.predict(data)

    return {"prediction": prediction.tolist()}


@app.get("/get_predict")
def get_predict(age: float, pclass: int, sex: str) -> dict:
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
    To make a GET request with query parameters, you can use the following curl command:
        curl -X 'GET' \
        'http://127.0.0.1:8000/get_predict?age=30&pclass=3&sex=male' \
        -H 'accept: application/json'

    """
    # Convert input data to DataFrame
    data_dict = {"age": age, "pclass": pclass, "sex": sex}
    data = pd.DataFrame([data_dict])

    prediction = model.predict(data)

    return {"prediction": prediction.tolist()}
