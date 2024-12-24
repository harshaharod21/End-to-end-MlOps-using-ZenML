from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model config

    """
    name_model:str="LinearRegression"