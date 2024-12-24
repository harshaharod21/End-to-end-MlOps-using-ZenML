from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__=="__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    train_pipeline(data_path="C:\All_projects\ZenML_Mlops\data\olist_customers_dataset.csv")


#command to open the mlflow uri

#mlflow ui --backend-store-uri "file:C:\Users\harod\AppData\Roaming\zenml\local_stores\5668f4e5-e7a8-4dc4-9fb6-d5d03ee44cfe\mlruns"
