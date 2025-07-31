from typing import Any

import requests
import streamlit as st
from PIL import Image

# define url
url = "http://127.0.0.1:8000"
headers = {"accept": "application/json"}


def predict(input_data: dict) -> dict[Any, Any]:
    response = requests.post(f"{url}/post_predict", headers=headers, json=input_data, timeout=300)

    return dict(response.json())


st.title("Titanic Survival Prediction")
st.write("Predict whether a passenger survived the Titanic disaster based on their features.")

# load image
image = Image.open("src/app/images/titanic.jpg")
st.image(image, caption="Titanic Ship", width=200, use_container_width=True)

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, value=10)
pclass = st.number_input("Passenger Class (1, 2, or 3)", min_value=1, max_value=3, value=1)
sex = st.selectbox("sex", ("male", "female"))

data = {"age": age, "pclass": pclass, "sex": sex}


if st.button("Predict"):
    # Make prediction
    response = predict(data)
    prediction = response["prediction"]
    probability = response["probability"]

    st.write(response)

    # Display result
    if prediction == 1:
        st.success("The passenger is predicted to have survived.")
        st.write(f"Probability of survival: {probability:.2f}")
    else:
        st.error("The passenger is predicted not to have survived.")
        st.write(f"Probability of survival: {probability:.2f}")
