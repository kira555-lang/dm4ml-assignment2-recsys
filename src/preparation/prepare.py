import os
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PREPARATION] - %(levelname)s - %(message)s")

def run_preparation(config):
    raw_dir = config["paths"]["raw_dir"]
    proc_dir = config["paths"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    users_df = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    items_df = pd.read_csv(os.path.join(raw_dir, "items.csv"))
    inter_df = pd.read_csv(os.path.join(raw_dir, "validated_interactions.csv"))

    # Standardize formats & data types
    users_df["user_id"] = users_df["user_id"].astype(str).str.strip()
    items_df["item_id"] = items_df["item_id"].astype(str).str.strip()
    items_df["category"] = items_df["category"].astype(str).str.strip()

    inter_df["user_id"] = inter_df["user_id"].astype(str).str.strip()
    inter_df["item_id"] = inter_df["item_id"].astype(str).str.strip()
    inter_df["rating"] = inter_df["rating"].astype(float)
    inter_df["timestamp"] = pd.to_datetime(inter_df["timestamp"])

    # Deduplicate multiple ratings by the same user on the same item (keep latest interaction)
    inter_df = inter_df.sort_values("timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="last")

    # Chronological Time-Based Train/Test Split (80% historical, 20% future)
    split_q = config["preparation"]["split_quantile"]
    cutoff_time = inter_df["timestamp"].quantile(split_q)

    train_df = inter_df[inter_df["timestamp"] <= cutoff_time].copy()
    test_df = inter_df[inter_df["timestamp"] > cutoff_time].copy()

    # Save cleaned & split datasets in Parquet format
    users_df.to_parquet(os.path.join(proc_dir, "users_clean.parquet"), index=False)
    items_df.to_parquet(os.path.join(proc_dir, "items_clean.parquet"), index=False)
    train_df.to_parquet(os.path.join(proc_dir, "train_interactions.parquet"), index=False)
    test_df.to_parquet(os.path.join(proc_dir, "test_interactions.parquet"), index=False)

    logging.info(f"Time-based split @ {cutoff_time}: Train={len(train_df)} rows, Test={len(test_df)} rows.")
    return train_df, test_df

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_preparation(cfg)
