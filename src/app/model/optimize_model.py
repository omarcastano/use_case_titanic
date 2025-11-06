"""
Optimize the Titanic model using GridSearchCV.
"""

if __name__ == "__main__":
    from src.app.data.data import get_dataset
    from src.app.model.model import TitanicModel
    from src.app.model.optimize import ModelOptimizer

    # get data
    train, test = get_dataset()

    # define model
    model = TitanicModel()

    # define parameter grid
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__max_iter": [1000, 2000],
        "classifier__penalty": ["l1", "l2"],
    }

    optimizer = ModelOptimizer(model, param_grid, mlflow_uri="https://dagshub.com/omar.castano25/use_case_titanic.mlflow")
    optimizer.optimize(train.drop(columns=["survived"]), train["survived"].to_numpy())
    optimizer.save_model("src/app/model/trained_models/optimized_titanic_model.joblib")
