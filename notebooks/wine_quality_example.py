# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# %% Imports
import os
import warnings
import sys
import time
import pandas as pd
import numpy as np
import plotly

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn

from importlib import reload

import utils as u
from visualization_helpers import plot_residuals


# %% CONFIGURATION
warnings.filterwarnings("ignore")
np.random.seed(40)

# In the UI run_names are grouped under experiment names
# We will group the data as the experiment and modeling approach as the runs
# Note without a run_name MLFlow will makeup a name and without an Experiment Name MLFlow will use "Default"

experiment_name="Wine Quality"
run_name="elastic_net_intro"

experiment_id=u.get_or_create_experiment(experiment_name)

alpha = .5
l1_ratio = .5

# %% DATA PREP

# Read the wine-quality csv file
current_file_path=os.path.dirname(__file__)
wine_path = os.path.join(current_file_path,"../data", "wine-quality.csv")
data = pd.read_csv(wine_path)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train["quality"]
test_y = test["quality"]

# %% MLFLOW

# We don't want duplicate run names when the code is executed many times
u.clear_run(experiment_id,run_name)

with mlflow.start_run(run_name=run_name,experiment_id=experiment_id):
    
    # tags are optional but can be helpful if you have many experiments and you later
    # want to search and filter among them in the UI
    mlflow.set_tags(
        tags={
            "project": "Wine Quality",
            "model_family": "elastic_net",
            "feature_set_version": 1,
        }
    )
    
    lr = ElasticNet(alpha = alpha,l1_ratio = l1_ratio,random_state=42)
    train_start=time.time()
    lr.fit(train_x, train_y)
    train_finish=time.time()
    train_time_seconds=train_finish-train_start

    predicted_qualities = lr.predict(test_x)

    eval_metrics = u.evaluate_regression_metrics(test_y, predicted_qualities)
    
    # Logging numbers or objects saves both locally under the run folder sub directory and in the UI
    
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    
    # MLFlow can log any arbitrary user defined metric. For this purpose we will record
    # training time and object size. This is useful to quickly see if an approach is not computationally
    # feasible or to determine if any performance gains are worth the computational cost
    model_memory_bytes=sys.getsizeof(lr)
    mlflow.log_metric("train_time_seconds",train_time_seconds)
    mlflow.log_metric("model_memory_bytes",model_memory_bytes)
    
    # MLFlow parameters and metrics can also be entered in as a dictionary
    # See the singular VS plural version log_metric VS log_metrics, log_param VS log_params
    mlflow.log_metrics(eval_metrics)

    # MLFlow will log supplementary meta data (EX versioning, packages) alongside the model 
    mlflow.sklearn.log_model(lr, "model")
    
    # MLFlow can also log a a matplotlib figure
    residual_fig=plot_residuals(valid_y=test_y,preds=predicted_qualities)
    # model, metrics, params have dedicated sub folders
    # other types of logged objects need a specified sub folder as such we save under plots
    mlflow.log_figure(residual_fig,"plots/elastic_net_example_residuals.png")
    
    # MLflow can also log a plotly example. We will convert to plotly and log both side by side.
    residual_fig_plotly=plotly.tools.mpl_to_plotly(residual_fig)
    mlflow.log_figure(residual_fig_plotly, "plots/elastic_net_example_residuals_plotly.html")
    
    # MLFlow cannot directly log a data set. However it can upload most types of presaved file
    # many of which can be viewed directly in the UI
    # In this example we will calculate metrics by quality and save/log using various data types for comparison
    eval_data:pd.DataFrame=test_x.\
        assign(quality=test_y,pred=predicted_qualities)
    eval_data['error']=eval_data['quality'] - eval_data['pred']
    eval_data['absolute_error']=np.abs(eval_data['error'])
    eval_data['squared_error']=eval_data['error'].pow(2)
    metrics_by_quality = eval_data.\
                groupby("quality",as_index=False).\
                agg(mean_absolute_error=('absolute_error','mean'),
                    mean_squared_error=('squared_error','mean')
                    )
                
    extension_type_dataframe_save_method_name_dict = {
        "html":'to_html',
        "csv":'to_csv',
        "pkl":'to_pickle',
        "md":'to_markdown',
        "json":'to_json'
    }
    for extension_name, method_name in extension_type_dataframe_save_method_name_dict.items():
        save_func=getattr(metrics_by_quality, method_name)
        save_file=os.path.join(current_file_path,"../data_outputs/","metrics_by_quality." + extension_name)
        save_func(save_file)
        # Unlike previous logging, the inputs are the file itself rather than a python object
        mlflow.log_artifact(local_path=save_file,artifact_path="data")
        

    mlflow.end_run()


