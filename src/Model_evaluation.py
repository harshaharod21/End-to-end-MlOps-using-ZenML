import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
import numpy as np  #numpy nd array and not list or series
from sklearn.metrics import mean_squared_error,r2_score

#create evaluation strategy
 
class eval_startegy(ABC):

  """
Evaluation Strategy

  """
  @abstractmethod
  def calculate_scores(self,y_true: np.ndarray,y_pred:np.ndarray)-> float:
    pass
  

class MSE(eval_startegy):
  
  """Evaluation Strategy that will use RMSE"""

  def calculate_scores(self, y_true:np.ndarray, y_pred: np.ndarray)-> float:
    try:
      
      logging.info("Caluclating MSE")
      mse=mean_squared_error(y_true, y_pred)
      logging.info("MSE:{}".format(mse))
      return mse
    
    except Exception as e:
      logging.error("Error while calcuulating RMSE scores{}".format(e))
      raise e

class R2(eval_startegy):
  """Calculating R2 scores"""


  def calculate_scores(self,y_true:np.ndarray, y_pred: np.ndarray)-> float:
    try:
        logging.info("Calculating R2 score")
        r2=r2_score(y_true,y_pred)
        logging.info("R2 score:{}".format(r2))
        return r2
    except Exception as e:
      logging.error("Error in calculating r2 score {}".format(e))
      raise e
    



    
  

      

      
    
  
