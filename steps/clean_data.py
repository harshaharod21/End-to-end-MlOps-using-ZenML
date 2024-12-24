import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(df: pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and splits it into train and test

    Args: 
         df: raw data
    returns:
            X_train
            Y_train
            X_test
            Y_test

    """
    
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning= DataCleaning(df, process_strategy)
        processed_data=data_cleaning.handle_data()
        split_strategy= DataSplitStrategy()
        data_cleaning= DataCleaning(processed_data, split_strategy)
        X_train,X_test,y_train,y_test= data_cleaning.handle_data()
        return X_train,X_test,y_train,y_test
    
        logging.info("Data cleaning completed")

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e 



