import pandas as pd
import streamlit as st
from PIL import Image

from src.app.model.model import TitanicModel

# load trained model
model = TitanicModel()
model.load_model("src/app/model/trained_models/titanic_model.joblib")

st.title("Titanic Survival Prediction")
st.write("Predict whether a passenger survived the Titanic disaster based on their features.")

# load image
image = Image.open("src/app/images/titanic.jpg")
st.image(image, caption="Titanic Ship", width=200, use_container_width=True)

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, value=10)
pclass = st.number_input("Passenger Class (1, 2, or 3)", min_value=1, max_value=3, value=1)
sex = st.selectbox("sex", ("male", "female"))

input_data = {"age": age, "pclass": pclass, "sex": sex}
input_data = pd.DataFrame([input_data])

if st.button("Predict"):
    # Make prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    result = {"prediction": int(prediction[0]), "probabilities": round(probabilities[0], 2)}
    st.write(result)

    # Display result
    if prediction == 1:
        st.success("The passenger is predicted to have survived.")
        st.write(f"Probability of survival: {probabilities[0]:.2f}")
    else:
        st.error("The passenger is predicted not to have survived.")
        st.write(f"Probability of survival: {probabilities[0]:.2f}")
