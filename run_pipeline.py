import time
import yaml
import logging
from src.ingestion.ingest import generate_raw_datasets
from src.validation.validate import run_validation
from src.preparation.prepare import run_preparation
from src.transformation.transform import run_transformation
from src.features.feature_store import build_feature_store
from src.training.train import run_training
from src.evaluation.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s")

def execute_workflow():
    start_ts = time.time()
    logging.info("=================================================================")
    logging.info("STARTING END-TO-END RECOMMENDATION DATA MANAGEMENT PIPELINE")
    logging.info("=================================================================")

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    stages = [
        ("Stage 1: Data Ingestion", generate_raw_datasets),
        ("Stage 2: Data Validation & Quarantine", run_validation),
        ("Stage 3: Data Preparation & Temporal Split", run_preparation),
        ("Stage 4: Data Transformation & Matrix Creation", run_transformation),
        ("Stage 5: Feature Store Generation", build_feature_store),
        ("Stage 6: Model Training & MLflow Tracking", run_training),
        ("Stage 7: Model Evaluation & Reporting", run_evaluation),
    ]

    for stage_name, stage_fn in stages:
        s_t = time.time()
        logging.info(f"--> Executing {stage_name}...")
        stage_fn(config)
        logging.info(f"<-- Completed {stage_name} in {time.time() - s_t:.2f}s")

    total_time = time.time() - start_ts
    logging.info("=================================================================")
    logging.info(f"PIPELINE EXECUTION SUCCEEDED IN {total_time:.2f} SECONDS")
    logging.info("=================================================================")

if __name__ == "__main__":
    execute_workflow()
