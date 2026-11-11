import os
import pickle
import yaml
import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ensure model class definitions are importable for pickle
import sys
sys.path.append(os.getcwd())
from src.training.train import PopularityBaselineRecommender, SVDHybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EVALUATION] - %(levelname)s - %(message)s")

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return 0.0

def ndcg_at_k(recommended_items, relevant_items, k):
    r = [1.0 if item in relevant_items else 0.0 for item in recommended_items[:k]]
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(r, k) / idcg

def intra_list_diversity(recommended_items, item_cat_map):
    if len(recommended_items) <= 1:
        return 0.0
    cats = [item_cat_map.get(i, "Unknown") for i in recommended_items]
    n = len(cats)
    dissimilar_pairs = 0
    total_pairs = n * (n - 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            if cats[i] != cats[j]:
                dissimilar_pairs += 1
    return dissimilar_pairs / total_pairs

def evaluate_model(model, model_name, train_df, test_df, item_features, k=10, rel_threshold=3.5):
    # 1. Explicit Rating Prediction Metrics (RMSE & MAE)
    y_true = test_df["rating"].values
    y_pred = [model.predict_rating(u, i) for u, i in zip(test_df["user_id"], test_df["item_id"])]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 2. Top-K Ranking Metrics (Precision@K, Recall@K, NDCG@K, Coverage, Diversity)
    train_seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    relevant_test = (
        test_df[test_df["rating"] >= rel_threshold]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    all_catalog_items = set(item_features["item_id"].unique())
    item_cat_map = dict(zip(item_features["item_id"], item_features["category"]))

    precisions, recalls, ndcgs, diversities = [], [], [], []
    recommended_catalog_union = set()

    for user_id, rel_items in relevant_test.items():
        if not rel_items:
            continue
        seen = train_seen.get(user_id, set())
        recs = model.recommend(user_id, seen, k=k)
        recommended_catalog_union.update(recs)

        hits = len(set(recs[:k]) & rel_items)
        precisions.append(hits / k)
        recalls.append(hits / len(rel_items))
        ndcgs.append(ndcg_at_k(recs, rel_items, k))
        diversities.append(intra_list_diversity(recs[:k], item_cat_map))

    coverage = len(recommended_catalog_union) / len(all_catalog_items)

    return {
        "Model": model_name,
        "K": k,
        f"Precision@{k}": round(float(np.mean(precisions)), 4),
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
        "Coverage": round(float(coverage), 4),
        "Diversity": round(float(np.mean(diversities)), 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4)
    }

def run_evaluation(config):
    proc_dir = config["paths"]["processed_dir"]
    feat_dir = config["paths"]["features_dir"]
    models_dir = config["paths"]["models_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(proc_dir, "train_interactions.parquet"))
    test_df = pd.read_parquet(os.path.join(proc_dir, "test_interactions.parquet"))
    item_features = pd.read_parquet(os.path.join(feat_dir, "item_features.parquet"))

    with open(os.path.join(models_dir, "popularity_model.pkl"), "rb") as f:
        pop_model = pickle.load(f)
    with open(os.path.join(models_dir, "svd_hybrid_model.pkl"), "rb") as f:
        svd_model = pickle.load(f)

    mlflow.set_tracking_uri(config["training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["training"]["experiment_name"])

    results = []
    rel_thresh = config["evaluation"]["relevance_threshold"]

    for k in config["evaluation"]["k_values"]:
        res_pop = evaluate_model(pop_model, "Baseline Popularity", train_df, test_df, item_features, k=k, rel_threshold=rel_thresh)
        res_svd = evaluate_model(svd_model, "Personalized SVD Hybrid", train_df, test_df, item_features, k=k, rel_threshold=rel_thresh)
        results.extend([res_pop, res_svd])

        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"Eval_Comparison_K{k}"):
            for metric_key, val in res_pop.items():
                if isinstance(val, float):
                    mlflow.log_metric(f"Baseline_{metric_key.replace('@', '_at_')}", val)
            for metric_key, val in res_svd.items():
                if isinstance(val, float):
                    mlflow.log_metric(f"SVD_{metric_key.replace('@', '_at_')}", val)

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(reports_dir, "model_evaluation_comparison.csv")
    md_path = os.path.join(reports_dir, "model_evaluation_comparison.md")

    results_df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Model Evaluation Comparison Report\n\n")
        f.write(results_df.to_markdown(index=False))

    logging.info(f"Evaluation complete. Results saved to {csv_path}")
    print("\n" + results_df.to_string(index=False) + "\n")
    return results_df

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_evaluation(cfg)
