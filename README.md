# Example MLflow project
This repo includes an example of setting up MLFlow server locally. 

## Set Up Development Environment:
```
conda env create -f conda.yaml
conda activate mlflow_env
```

## Instructions
Run mlflow experiments manually:
```
mlflow ui -p 5050
python3 train.py # with a set of parameters
```
Or run mlflow experiments through MLProject
```
mlflow ui -p 5050
mlflow run . --env-manager=local 
```
You should see the UI by opening URL http://localhost:5050/