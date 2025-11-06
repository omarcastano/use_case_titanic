if __name__ == "__main__":
    from src.app.data.data import get_dataset
    from src.app.model.model import TitanicModel

    train, test = get_dataset()
    model = TitanicModel()
    model.fit(train.drop(columns=["survived"]), train["survived"].to_numpy())
    model.save_model("src/app/model/trained_models/titanic_model.joblib")
