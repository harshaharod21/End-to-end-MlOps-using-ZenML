import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker




@step(experiment_tracker=experiment_tracker.name)
def  train_model(X_train:pd.DataFrame,Y_train:pd.DataFrame,config:ModelNameConfig)-> RegressorMixin:
    """
    Train the mdoel on the ingested data
    Args:
         df: the ingested data
    """

    try:
        if config.name_model == "LinearRegression":
          mlflow.sklearn.autolog()        #To log our model,scores
          
          model_train=LinearRegressionModel()
          model=model_train.handle_dataset(X_train,Y_train)
          return model
        else:
            raise ValueError("Model  {}  not supported".format(config.model_name))

    except Exception as e:
          logging.error("Erro while model development{}".format(e))
          raise e



        
    

