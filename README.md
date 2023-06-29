# Example MLflow project
This repo includes an example of setting up MLFlow server locally. 

## Set Up Development Environment:
```
poetry install
poetry shell
```

## Instructions
Run mlflow experiments manually:
```
python3 train.py # with a set of parameters
mlflow ui
```
Or run mlflow experiments through MLProject
```
mlflow run . --env-manager=local 
mlflow ui
```
You should see the UI by opening URL http://localhost:5050/