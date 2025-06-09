if __name__ == "__main__":
    from src.app.data.data import get_dataset
    from src.app.model.model import TitanicModel

    data = get_dataset()
    model = TitanicModel()
    model.fit(data.drop(columns=["survived"]), data["survived"].to_numpy())
    model.save_model("src/app/model/trained_models/titanic_model.joblib")
