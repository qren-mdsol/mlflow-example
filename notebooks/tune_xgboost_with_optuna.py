
# %% Imports
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import yaml
import mlflow
import optuna
import os

from importlib import reload
import utils as u
from visualization_helpers import plot_residuals, plot_xgb_feature_importance

# %% Set Experiment Configuration
current_file_path=os.path.dirname(__file__)
experiment_id = u.get_or_create_experiment("Apples Demand")
mlflow.set_experiment(experiment_id=experiment_id)
run_name = "xgboost_optuna"

with open("config/xgboost_optuna.yaml") as f:
    xgb_config = yaml.load(f,Loader=yaml.FullLoader)

# %% Preprocess Data
df=pd.read_csv("data/apple-sales.csv")

X = df.drop(columns=["date", "demand"])
y = df["demand"]
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)

# %% Specify Objective function

def objective(trial:optuna.Trial):

    # Define hyperparameters
    search_grid=xgb_config['search_grid']
    params = {
        "objective": xgb_config['objective'],
        "eval_metric": xgb_config['eval_metric'],
        "booster": trial.suggest_categorical("booster", search_grid['booster']),
        "lambda": trial.suggest_float("lambda", 
                                      *search_grid['lambda_range'],
                                      log=True),
        "alpha": trial.suggest_float("alpha", *search_grid['alpha_range'],log=True),
    }
    if params['booster'] in ['gbtree','dart']:
        params["max_depth"] = trial.suggest_int("max_depth",*search_grid['max_depth_range'])
        params["eta"] = trial.suggest_float("eta",
                                            search_grid['eta_range'][0],
                                            search_grid['eta_range'][1], 
                                            log=True)
        params["gamma"] = trial.suggest_float("gamma",
                                              *search_grid['gamma_range'],
                                                log=True)
        params["grow_policy"] = trial.suggest_categorical(
            "grow_policy", search_grid['grow_policy']
        )
    child_run_name=u.convert_params_to_string(params)
    with mlflow.start_run(run_name=child_run_name,nested=True):
        # Train XGBoost model
        train_start=time.time()
        bst = xgb.train(params, dtrain)
        train_finish=time.time()
        train_time_seconds=train_finish-train_start
        
        preds = bst.predict(dvalid)
        eval_metrics=u.evaluate_regression_metrics(valid_y,preds)
        error_score=eval_metrics['rmse']
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(eval_metrics)
        mlflow.log_metric("train_time_seconds", train_time_seconds)

        return error_score



# %% Tune and Retrain Best Params

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    # Initialize the Optuna study
    study = optuna.create_study(direction="minimize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    study.optimize(objective, n_trials=xgb_config['ntrial'])

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_rmse", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Apple Demand Project",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        }
    )

    # retrain
    model = xgb.train(study.best_params, dtrain)
    preds=model.predict(dvalid)
    eval_metrics=u.evaluate_regression_metrics(valid_y,preds)
    mlflow.log_metrics(eval_metrics)

    # Log the feature importances plot
    importances = plot_xgb_feature_importance(model, booster=study.best_params.get("booster"))
    mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

    # Log the residuals plot
    residuals = plot_residuals(valid_y,preds)
    mlflow.log_figure(figure=residuals, artifact_file="residuals.png")

    artifact_path = "xgboost_optuna_model"

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path=artifact_path,
        input_example=train_x.iloc[[0]],
        model_format="ubj",
        metadata={"model_data_version": 1},
    )
    model_uri = mlflow.get_artifact_uri(artifact_path)
    
    mlflow.end_run()

# %% Predict From Loaded Model
loaded = mlflow.xgboost.load_model(model_uri)    
batch_dmatrix = xgb.DMatrix(X)
inference = loaded.predict(batch_dmatrix)
infer_df = df.copy()
infer_df["predicted_demand"] = inference
infer_df.head()
 