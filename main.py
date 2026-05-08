# entry point for the ML Churn Detection project
from src.pipelines.training_pipeline import run_training_pipeline

# run the full training pipeline on the raw dataset
run_training_pipeline("data/raw/Telco_Customer_Churn.csv")