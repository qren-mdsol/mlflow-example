# Example MLflow project
This repo includes an example of setting up MLFlow server locally. 

## Set Up Development Environment:
```
poetry install
poetry shell
```

## Instructions



Their are scripts that are meant to be executed on the terminal as well as notebooks. For scripts run on the main terminal via

```
python3 scripts/[name of python script]
```

On a seperate terminal call

```
mlflow ui -p 5050
```
Copy the link to your browser.

There are also notebooks for interactive experimentation. For users that prefer not to use notebooks, The notebooks have an interactive python script version using # %% synced via jupytext.

The python path will need to be configured to root For VSCode users this can be done in the workspace settings.json file as follows: 

```
"terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}/"
    },
"jupyter.notebookFileRoot": "${workspaceFolder}/"
```



