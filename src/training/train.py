import os
import pickle
import yaml
import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TRAINING] - %(levelname)s - %(message)s")

class PopularityBaselineRecommender:
    """Recommends top items ranked by Bayesian popularity score and predicts item mean rating."""
    def __init__(self):
        self.item_scores = {}
        self.ranked_items = []
        self.global_mean = 3.5

    def fit(self, train_df, item_features):
        self.global_mean = float(train_df["rating"].mean())
        sorted_items = item_features.sort_values("bayesian_popularity_score", ascending=False)
        self.ranked_items = sorted_items["item_id"].tolist()
        self.item_scores = dict(zip(sorted_items["item_id"], sorted_items["bayesian_popularity_score"]))
        return self

    def predict_rating(self, user_id, item_id):
        return float(np.clip(self.item_scores.get(item_id, self.global_mean), 1.0, 5.0))

    def recommend(self, user_id, seen_items, k=10):
        recs = [item for item in self.ranked_items if item not in seen_items]
        return recs[:k]


class SVDHybridRecommender:
    """Matrix Factorization (TruncatedSVD) over mean-centered User-Item ratings + Category Affinity Boost."""
    def __init__(self, n_components=24, n_iter=15, random_state=42, affinity_boost=0.15):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.affinity_boost = affinity_boost
        self.global_mean = 3.5
        self.user_Index = {}
        self.item_Index = {}
        self.item_ids = []
        self.user_bias = {}
        self.item_bias = {}
        self.pred_matrix = None
        self.user_pref_cat = {}
        self.item_cat = {}
        self.fallback_items = []

    def fit(self, train_df, user_features, item_features):
        self.global_mean = float(train_df["rating"].mean())
        self.user_bias = dict(zip(user_features["user_id"], user_features["avg_rating"] - self.global_mean))
        self.item_bias = dict(zip(item_features["item_id"], item_features["item_avg_rating"] - self.global_mean))
        self.user_pref_cat = dict(zip(user_features["user_id"], user_features["preferred_category"]))
        self.item_cat = dict(zip(item_features["item_id"], item_features["category"]))
        self.fallback_items = item_features.sort_values("bayesian_popularity_score", ascending=False)["item_id"].tolist()

        pivot = train_df.pivot(index="user_id", columns="item_id", values="rating")
        self.user_Index = {u: idx for idx, u in enumerate(pivot.index)}
        self.item_Index = {m: idx for idx, m in enumerate(pivot.columns)}
        self.item_ids = list(pivot.columns)

        # Center ratings by user mean before SVD
        user_means = pivot.mean(axis=1)
        centered_matrix = pivot.sub(user_means, axis=0).fillna(0.0).values

        n_comp = min(self.n_components, min(centered_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, n_iter=self.n_iter, random_state=self.random_state)
        U = svd.fit_transform(centered_matrix)
        VT = svd.components_

        reconstructed = np.dot(U, VT) + user_means.values[:, np.newaxis]
        self.pred_matrix = np.clip(reconstructed, 1.0, 5.0)
        return self

    def predict_rating(self, user_id, item_id):
        if user_id in self.user_Index and item_id in self.item_Index:
            u_idx = self.user_Index[user_id]
            i_idx = self.item_Index[item_id]
            base_pred = float(self.pred_matrix[u_idx, i_idx])
        else:
            u_b = self.user_bias.get(user_id, 0.0)
            i_b = self.item_bias.get(item_id, 0.0)
            base_pred = self.global_mean + u_b + i_b

        if self.user_pref_cat.get(user_id) == self.item_cat.get(item_id):
            base_pred += self.affinity_boost
        return float(np.clip(base_pred, 1.0, 5.0))

    def recommend(self, user_id, seen_items, k=10):
        if user_id not in self.user_Index:
            return [m for m in self.fallback_items if m not in seen_items][:k]

        u_idx = self.user_Index[user_id]
        scores = self.pred_matrix[u_idx].copy()
        pref_c = self.user_pref_cat.get(user_id)

        scored_candidates = []
        for i_idx, item_id in enumerate(self.item_ids):
            if item_id in seen_items:
                continue
            s = float(scores[i_idx])
            if pref_c and self.item_cat.get(item_id) == pref_c:
                s += self.affinity_boost
            scored_candidates.append((item_id, s))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scored_candidates[:k]]


def run_training(config):
    proc_dir = config["paths"]["processed_dir"]
    feat_dir = config["paths"]["features_dir"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(proc_dir, "train_interactions.parquet"))
    user_features = pd.read_parquet(os.path.join(feat_dir, "user_features.parquet"))
    item_features = pd.read_parquet(os.path.join(feat_dir, "item_features.parquet"))

    mlflow.set_tracking_uri(config["training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["training"]["experiment_name"])

    # 1. Train Baseline Popularity Recommender
    with mlflow.start_run(run_name="Baseline_Popularity_Model"):
        mlflow.log_param("model_type", "Bayesian_Popularity_Baseline")
        mlflow.log_param("damping_factor_m", 10.0)
        pop_model = PopularityBaselineRecommender().fit(train_df, item_features)
        pop_path = os.path.join(models_dir, "popularity_model.pkl")
        with open(pop_path, "wb") as f:
            pickle.dump(pop_model, f)
        mlflow.log_artifact(pop_path)
        logging.info("Trained & logged Popularity Baseline Model.")

    # 2. Train Personalized SVD + Feature Hybrid Recommender
    with mlflow.start_run(run_name="Personalized_SVD_Hybrid_Model"):
        n_comp = config["training"]["svd_components"]
        n_iter = config["training"]["svd_iterations"]
        mlflow.log_param("model_type", "TruncatedSVD_Hybrid")
        mlflow.log_param("n_components", n_comp)
        mlflow.log_param("n_iterations", n_iter)
        mlflow.log_param("affinity_boost", 0.15)

        svd_model = SVDHybridRecommender(
            n_components=n_comp, n_iter=n_iter, random_state=config["random_seed"]
        ).fit(train_df, user_features, item_features)

        svd_path = os.path.join(models_dir, "svd_hybrid_model.pkl")
        with open(svd_path, "wb") as f:
            pickle.dump(svd_model, f)
        mlflow.log_artifact(svd_path)
        logging.info("Trained & logged Personalized SVD Hybrid Model.")

    return pop_model, svd_model

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_training(cfg)
