import logging
import pandas as pd
from zenml import step
from src.Model_evaluation import MSE,R2
from sklearn.base import RegressorMixin 
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker=experiment_tracker.name)

def evaluate(model:RegressorMixin,X_test:pd.DataFrame,Y_test:pd.Series)-> Tuple[
    Annotated[float,"r2_score"],
    Annotated[float,"rmse_score"],
] :
    """
    Evaluate the model on the ingested data
    Args:
         df: the ingested data
    """
    try:
        prediction = model.predict(X_test)

        rmse_obj=MSE()
        rmse_score=rmse_obj.calculate_scores(prediction,Y_test)
        mlflow.log_metric("mse",rmse_score)
        r2_obj=R2()
        r2_score=r2_obj.calculate_scores(prediction,Y_test)
        mlflow.log_metric("r2_score",r2_score)

        return rmse_score,r2_score
    except Exception as e:
        logging.error("Error while evaluation {}".format(e))
        raise e
