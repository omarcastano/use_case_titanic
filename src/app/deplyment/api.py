import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# load trained model
from src.app.model.model import TitanicModel

model = TitanicModel()
model.load_model("src/app/model/trained_models/titanic_model.joblib")

app = FastAPI()


class InputData(BaseModel):
    sex: str = "male"
    pclass: int = 3
    age: float = 30.0


@app.post("/predict")
def predict(data: InputData) -> dict:
    """Endpoint to make predictions using the Titanic survival model.
    Arguments:
    ----------
    data : InputData
        The input data containing features for prediction.
    Returns:
    -------
    dict
        A dictionary containing the prediction result.
    """
    # Convert input data to DataFrame
    data_dict = data.dict()
    data = pd.DataFrame([data_dict])

    prediction = model.predict(data)

    return {"prediction": prediction.tolist()}
