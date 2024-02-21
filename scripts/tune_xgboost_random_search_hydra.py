# %% Imports
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import yaml
import mlflow
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import utils as u
from visualization_helpers import plot_residuals, plot_xgb_feature_importance

from importlib import reload

print(os.getcwd())

# %% Main

@hydra.main(version_base=None, config_path="../config", config_name="xgboost_random_search")
def tune_xgb_random_search(cfg : DictConfig) -> None:
    # %% Set Experiment Configuration
    experiment_id = u.get_or_create_experiment("Apples Demand")
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = cfg['run_name']
        
    # %% Preprocess Data
    current_filepath=os.path.dirname(__file__)
    apple_sales_df=pd.read_csv(os.path.join(current_filepath,"../data/apple-sales.csv"))

    X = apple_sales_df.drop(columns=["date", "demand"])
    y = apple_sales_df["demand"]
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    # %% MLFlow

    param_grid=u.expand_grid(**cfg['search_grid'])
    param_grid_sampled:pd.DataFrame=param_grid.sample(cfg['nsample'])    

    # Prevent flooding MLflow with duplicate run names upon re-execution of the code
    u.clear_run(experiment_id,run_name)
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
        # We will optimize on rmse
        rmse_results=[]
        
        for row_index, row in param_grid_sampled.iterrows():
            # row_index=0; row=param_grid_sampled.iloc[0]
            params=row.to_dict()
            # the names that appear reflects the parameter
            child_run_name=u.convert_params_to_string(params)
            params['max_depth']=int(params['max_depth']) #dictionary conversion coerces to float
            params['objective']=cfg['objective']
            params['eval_metric']=cfg['eval_metric']
                
            # innitiate the child run, INSIDE the for loop of the params
            with mlflow.start_run(run_name=child_run_name,nested=True): 
                # Train XGBoost model
                train_start=time.time()
                bst = xgb.train(params, dtrain)
                train_finish=time.time()
                train_time_seconds=train_finish - train_start
                 
                preds = bst.predict(dvalid)
                eval_metrics=u.evaluate_regression_metrics(valid_y, preds)
                rmse_results.append(eval_metrics['rmse'])
                
                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(eval_metrics)
                mlflow.log_metric("train_time_seconds", train_time_seconds)
        
        # Retrain best model using rmse and visualize results
        param_grid_sampled['rmse']=rmse_results
        indx_min = param_grid_sampled['rmse'].idxmin()
        best_rmse=param_grid_sampled['rmse'].loc[indx_min]
        best_params=param_grid_sampled.\
            drop(columns=['rmse']).\
            loc[indx_min].\
            to_dict()
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_rmse", best_rmse)
        
        # Log tags to more easily search experiments
        mlflow.set_tags(
            tags={
                "project": "Apple Demand Project",
                "optimizer_engine": "random search",
                "model_family": "xgboost",
                "feature_set_version": 1,
            }
        )
        
        # Log a fit model instance
        model = xgb.train(best_params, dtrain)
        # Log Metrics
        preds=model.predict(dvalid)
        eval_metrics=u.evaluate_regression_metrics(actual=valid_y,pred=preds)
        mlflow.log_metrics(eval_metrics)

        # Log the feature importances plot
        importances = plot_xgb_feature_importance(model, booster=best_params['booster'])
        mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

        # Log the residuals plot
        residuals = plot_residuals(valid_y,preds)
        mlflow.log_figure(figure=residuals, artifact_file="residuals.png")
        
        # artifact path helps access the model object outside
        artifact_path = "xgb_random_search_model"
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=artifact_path,
            input_example=train_x.iloc[[0]],
            model_format="ubj",
            metadata={"model_data_version": 1},
        )
        
        mlflow.end_run()

if __name__ == "__main__":
    tune_xgb_random_search()

