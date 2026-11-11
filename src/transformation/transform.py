import os
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TRANSFORMATION] - %(levelname)s - %(message)s")

def run_transformation(config):
    proc_dir = config["paths"]["processed_dir"]
    train_df = pd.read_parquet(os.path.join(proc_dir, "train_interactions.parquet"))
    items_df = pd.read_parquet(os.path.join(proc_dir, "items_clean.parquet"))

    # 1. User Activity Summary
    user_summary = train_df.groupby("user_id").agg(
        activity_count=("interaction_id", "count"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        last_interaction_ts=("timestamp", "max")
    ).reset_index()
    user_summary["rating_std"] = user_summary["rating_std"].fillna(0.0)

    # 2. Item Popularity Summary
    item_summary = train_df.groupby("item_id").agg(
        interaction_count=("interaction_id", "count"),
        item_avg_rating=("rating", "mean")
    ).reset_index()
    item_summary = items_df[["item_id", "title", "category", "release_year"]].merge(
        item_summary, on="item_id", how="left"
    )
    item_summary["interaction_count"] = item_summary["interaction_count"].fillna(0).astype(int)
    global_mean = train_df["rating"].mean()
    item_summary["item_avg_rating"] = item_summary["item_avg_rating"].fillna(global_mean)

    # Bayesian damped popularity score: (v / (v + m)) * R + (m / (v + m)) * C
    m = 10.0
    v = item_summary["interaction_count"]
    R = item_summary["item_avg_rating"]
    item_summary["bayesian_popularity_score"] = (v / (v + m)) * R + (m / (v + m)) * global_mean

    # 3. User-Item Interaction Matrix (Explicit ratings pivot)
    interaction_matrix = train_df.pivot(index="user_id", columns="item_id", values="rating").fillna(0.0)

    user_summary.to_parquet(os.path.join(proc_dir, "user_activity_summary.parquet"), index=False)
    item_summary.to_parquet(os.path.join(proc_dir, "item_popularity_summary.parquet"), index=False)
    interaction_matrix.to_parquet(os.path.join(proc_dir, "user_item_matrix.parquet"))

    logging.info(f"Derived datasets created. Interaction matrix shape: {interaction_matrix.shape}")
    return user_summary, item_summary, interaction_matrix

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_transformation(cfg)
