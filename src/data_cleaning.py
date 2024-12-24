
# Here we are using strategy pattern to implement data cleaning

import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#abstarct class to define strategy

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    object creation
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)-> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):

    """
    Strategy for preprocessing data

    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        """
        Preprocess data

        """
        try:
          
            
        #other preprocessing steps add

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace= True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["payment_sequential"].fillna(data["payment_sequential"].median(), inplace=True)
            data["payment_installments"].fillna(data["payment_installments"].median(), inplace=True)
            data["payment_value"].fillna(data["payment_value"].median(), inplace=True)
            data["price"].fillna(data["price"].median(), inplace=True)
            


            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data

            
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

class DataSplitStrategy(DataStrategy):
    """
    Strategy for splitting the data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and split

        """
        try:
            X=data.drop(["review_score"],axis=1)
            Y=data["review_score"]
            X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2, random_state=42)
            return X_train, X_test,Y_train,Y_test
        except Exception as e:
            logging.error("Error in splitting dataset:{}".format(e))
            raise e

class DataCleaning:

    """
    Class for cleaning data which processes the data and divides it into train and test

    """
    def __init__(self, data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self)-> Union[pd.DataFrame,pd.Series]:
        """
        Handle data

        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
        



    


       
        




