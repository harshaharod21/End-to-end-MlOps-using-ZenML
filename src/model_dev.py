import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


#Creating model strategy

class ModelStrategy(ABC):

   """
   Abstract Class to create model development strategy

   """
   @abstractmethod
   def handle_dataset(self, X_train,Y_train):
      
      """Trains the model on the given data."""
      pass
   
     
    
class LinearRegressionModel(ModelStrategy):
   """
   Linear Regression Model Training

   """

   def handle_dataset(self, X_train,Y_train, **kwargs):
      """
      Defining model and fitting it
      """

      try:     
         
          model=LinearRegression(**kwargs)
          model.fit(X_train,Y_train)
          logging.info("Model training complete")
          
          return model 
      
      except Exception as e:
         logging.error("Error while training the model{}".format(e))
         raise e
      


      
   




      
   


    

   
      
