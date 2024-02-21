
# %% Imports
import pandas as pd
import numpy as np
import mlflow
from itertools import product
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% Mlflow
def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
def clear_run(experiment_id,run_name):
    """
    If a run name exists within the experiment ID delete it to enable a fresh start along with it's
    children runs. 
    """
    runs_df=mlflow.search_runs([experiment_id])
    
    if len(runs_df) == 0:
        return
    indx_run_name=runs_df['tags.mlflow.runName'] == run_name
    runs_df_w_name=runs_df[indx_run_name]
    for run_id in runs_df_w_name['run_id']:
            mlflow.delete_run(run_id)
            # if their are no children,column may not exist
            if 'tags.mlflow.parentRunId' in runs_df.columns:
                indx_child_id=runs_df['tags.mlflow.parentRunId'] == run_id
                children_ids=runs_df[indx_child_id]['run_id']
                for child_id in children_ids:
                    mlflow.delete_run(child_id)
                

# %% Optuna    
def champion_callback(study:optuna.Study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

# %% Tuning
def expand_grid(**kwargs)->pd.DataFrame:
    """
    Expands every combination of iterables into a DataFrame. Used for testing many combinations
    of paramaters and for other programming applications. Takes in column_name=iterable, column_name2=iterable, ...
    """    

    names = list(kwargs.keys())
    values = list(kwargs.values())
    grid= pd.DataFrame(list(product(*values)), columns = names)
    
    return grid

def convert_params_to_string(params: dict,sep: str="_",nround=3) -> str:
    """
    Create a string naming the params seperated by sep
    """
    out_string=''
    for param_index, param_name in enumerate(params):
        requires_sep_between_params=(param_index +1 ) != len(params)
        param_value=params[param_name]
        if type(param_value) == float:
            param_value=round(param_value,nround)
        out_string = out_string + param_name + sep + str(param_value)
        if requires_sep_between_params:
            out_string=out_string + sep
    return out_string


def evaluate_regression_metrics(actual, pred) -> dict:
    metric_dict = dict()
    metric_dict['rmse']=np.sqrt(mean_squared_error(actual, pred))
    metric_dict['mae'] = mean_absolute_error(actual, pred)
    metric_dict['r2'] = r2_score(actual, pred)
    
    return metric_dict


    