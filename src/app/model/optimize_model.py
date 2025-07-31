"""
Optimize the Titanic model using GridSearchCV.
"""

if __name__ == "__main__":
    from src.app.data.data import get_dataset
    from src.app.model.model import TitanicModel
    from src.app.model.optimize import ModelOptimizer

    # get data
    data = get_dataset()

    # define model
    model = TitanicModel()

    # define parameter grid
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__max_iter": [1000, 2000],
        "classifier__penalty": ["l1", "l2"],
    }

    optimizer = ModelOptimizer(model, param_grid, mlflow_uri="http://127.0.0.1:5000")
    optimizer.optimize(data.drop(columns=["survived"]), data["survived"].to_numpy())
    optimizer.save_model("src/app/model/trained_models/optimized_titanic_model.joblib")
