import os
from typing import NamedTuple

import numpy as np

import kfp.components as comp
from kfp import dsl, compiler,  Client

from client import create_client

# Define the data loading function
def load_data_op(random_state:int, test_size:float, 
    train_path: comp.OutputPath('CSV'),
    test_path: comp.OutputPath('CSV')):

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Load the Boston housing dataset
    X, Y = make_regression()
    data = pd.DataFrame(X)
    data['Y'] = Y
    # Split the data into training and test sets
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    # Save to output paths
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

# Define the model training function
def train_model_op(train_path: comp.InputPath('CSV'), model_path: comp.OutputPath('PKL')):# -> NamedTuple('Outputs', [('model', object), ('model_path', str)]):#, learning_rate: float, max_depth: int, subsample: float, n_estimators: int):
    import pickle
    import os
    import xgboost as xgb
    import pandas as pd

    train_data = pd.read_csv(train_path)
    
    # Create the XGBoost model
    xgb_model = xgb.XGBRegressor()

    # Fit the model to the training data
    xgb_model.fit(train_data.drop('Y', axis = 1), train_data['Y'])
    
    # Save model
    pickle.dump(xgb_model, open(model_path, "wb"))

# Define the model evaluation function
def evaluate_model_op(test_path: comp.InputPath('CSV'), model_path: comp.InputPath('PKL')) -> NamedTuple('Outputs', [
  ('mlpipeline_metrics', 'Metrics'),
]):
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import pickle
    import json

    test_data = pd.read_csv(test_path)
    xgb_model = pickle.load(open(model_path, "rb"))
    # Make predictions on the test set
    y_pred = xgb_model.predict(test_data.drop('Y', axis = 1))

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(test_data['Y'], y_pred)
    metrics = {
        'metrics': [{
        'name': 'mse-score', # The name of the metric. Visualized as the column name in the runs table.
        'numberValue':  mse, # The value of the metric. Must be a numeric value.
        'format': "RAW",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
    }
    return [json.dumps(metrics)]

def create_components():
    data_op = comp.create_component_from_func(func=load_data_op, base_image='huanjason/scikit-learn', output_component_file='load_data_component.yaml')
    train_op = comp.create_component_from_func(func=train_model_op, base_image='huanjason/scikit-learn', output_component_file='train_model_component.yaml')
    eval_op = comp.create_component_from_func(func=evaluate_model_op, base_image='huanjason/scikit-learn', output_component_file='evaluate_model_component.yaml')
    return data_op, train_op, eval_op

# Define the pipeline
@dsl.pipeline(
    name="XGBoost Pipeline",
    description="A pipeline that trains an XGBoost model on the Boston housing dataset."
)
def xgboost_pipeline(random_state: int =20, test_size: float =0.2):
    data_op, train_op, eval_op=create_components()
    data_prep = data_op(random_state, test_size)
    trainer = train_op(data_prep.outputs['train'])
    mse_score = eval_op(data_prep.outputs['test'], trainer.outputs['model']).output

if __name__ == '__main__':
    client = create_client()
    client.create_run_from_pipeline_func(xgboost_pipeline, {}, experiment_name="Test XG_Boost")
