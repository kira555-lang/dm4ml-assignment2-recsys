import os
import sys
import yaml
from datetime import datetime, timedelta

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.ingestion.ingest import generate_raw_datasets
from src.validation.validate import run_validation
from src.preparation.prepare import run_preparation
from src.transformation.transform import run_transformation
from src.features.feature_store import build_feature_store
from src.training.train import run_training
from src.evaluation.evaluate import run_evaluation

def _load_cfg():
    os.chdir(PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, "configs/config.yaml"), "r") as f:
        return yaml.safe_load(f)

default_args = {
    "owner": "ravi_pothana",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="end_to_end_recommendation_pipeline",
    default_args=default_args,
    description="End-to-End Data Management Pipeline for Recommendation System (DM4ML Assignment 2)",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["dm4ml", "recsys", "data-management"],
) as dag:

    t1_ingest = PythonOperator(
        task_id="1_data_ingestion",
        python_callable=lambda: generate_raw_datasets(_load_cfg()),
    )

    t2_validate = PythonOperator(
        task_id="2_data_validation",
        python_callable=lambda: run_validation(_load_cfg()),
    )

    t3_prepare = PythonOperator(
        task_id="3_data_preparation",
        python_callable=lambda: run_preparation(_load_cfg()),
    )

    t4_transform = PythonOperator(
        task_id="4_data_transformation",
        python_callable=lambda: run_transformation(_load_cfg()),
    )

    t5_features = PythonOperator(
        task_id="5_feature_management",
        python_callable=lambda: build_feature_store(_load_cfg()),
    )

    t6_train = PythonOperator(
        task_id="6_model_training_mlflow",
        python_callable=lambda: run_training(_load_cfg()),
    )

    t7_evaluate = PythonOperator(
        task_id="7_model_evaluation",
        python_callable=lambda: run_evaluation(_load_cfg()),
    )

    # Strict Sequential Dependency Chain
    t1_ingest >> t2_validate >> t3_prepare >> t4_transform >> t5_features >> t6_train >> t7_evaluate
