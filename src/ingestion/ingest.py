import os
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [INGESTION] - %(levelname)s - %(message)s")

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_raw_datasets(config):
    np.random.seed(config["random_seed"])
    raw_dir = config["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    n_users = config["ingestion"]["num_users"]
    n_items = config["ingestion"]["num_items"]
    n_inter = config["ingestion"]["num_interactions"]

    # 1. Generate Users Dataset
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    locations = ["IN-TG", "US-CA", "UK-LON", "DE-BE", "SG-SG", "CA-ON"]
    users_df = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(1, n_users + 1)],
        "age_group": np.random.choice(age_groups, n_users, p=[0.25, 0.35, 0.20, 0.12, 0.08]),
        "location": np.random.choice(locations, n_users),
        "signup_date": [
            (datetime(2024, 1, 1) + timedelta(days=int(x))).strftime("%Y-%m-%d")
            for x in np.random.randint(0, 365, n_users)
        ]
    })

    # 2. Generate Items Dataset
    genres = ["Sci-Fi", "Action", "Drama", "Thriller", "Comedy", "Documentary", "Noir", "Fantasy"]
    items_df = pd.DataFrame({
        "item_id": [f"M{i:04d}" for i in range(1, n_items + 1)],
        "title": [f"Cinema_Title_{i:04d}" for i in range(1, n_items + 1)],
        "category": np.random.choice(genres, n_items, p=[0.18, 0.18, 0.16, 0.15, 0.13, 0.08, 0.06, 0.06]),
        "release_year": np.random.randint(1995, 2026, n_items),
        "base_popularity": np.random.beta(2, 5, n_items)
    })

    # Latent preference structure for realistic signal
    user_genre_pref = {u: np.random.choice(genres, 2, replace=False) for u in users_df["user_id"]}
    item_genre_map = dict(zip(items_df["item_id"], items_df["category"]))
    item_pop_weights = items_df["base_popularity"].values / items_df["base_popularity"].sum()

    # 3. Generate User-Item Interactions
    sampled_users = np.random.choice(users_df["user_id"], n_inter)
    sampled_items = np.random.choice(items_df["item_id"], n_inter, p=item_pop_weights)

    ratings = []
    for u, m in zip(sampled_users, sampled_items):
        base_r = 3.0
        if item_genre_map[m] in user_genre_pref[u]:
            base_r += 1.2
        noise = np.random.normal(0, 0.6)
        r = np.clip(np.round((base_r + noise) * 2) / 2, 1.0, 5.0)
        ratings.append(r)

    base_time = datetime(2025, 1, 1)
    timestamps = [
        (base_time + timedelta(seconds=int(x))).strftime("%Y-%m-%d %H:%M:%S")
        for x in np.random.randint(0, 365 * 86400, n_inter)
    ]

    interactions_df = pd.DataFrame({
        "interaction_id": [f"INT{i:06d}" for i in range(1, n_inter + 1)],
        "user_id": sampled_users,
        "item_id": sampled_items,
        "rating": ratings,
        "interaction_type": np.random.choice(["watch", "rating", "completion"], n_inter, p=[0.5, 0.3, 0.2]),
        "timestamp": timestamps
    })

    # 4. Inject Controlled Data Quality Anomalies for Validation Testing
    n_anomalies = int(n_inter * config["ingestion"]["anomaly_rate"])
    anomaly_indices = np.random.choice(interactions_df.index, n_anomalies, replace=False)
    splits = np.array_split(anomaly_indices, 5)

    # Anomaly A: Missing user_id or rating
    interactions_df.loc[splits[0], "user_id"] = None
    # Anomaly B: Out-of-range ratings (> 5.0 or < 1.0)
    interactions_df.loc[splits[1], "rating"] = np.random.choice([-2.0, 0.0, 6.5, 9.9], len(splits[1]))
    # Anomaly C: Future / corrupted timestamps
    interactions_df.loc[splits[2], "timestamp"] = "2099-99-99 25:61:00"
    # Anomaly D: Referential integrity violation (Ghost user_id / item_id)
    interactions_df.loc[splits[3], "item_id"] = "M9999_GHOST"
    # Anomaly E: Exact duplicate rows
    dup_rows = interactions_df.loc[splits[4]].copy()
    interactions_df = pd.concat([interactions_df, dup_rows], ignore_index=True)

    # Drop temporary helper column from items
    items_df = items_df.drop(columns=["base_popularity"])

    # Save to CSV
    users_df.to_csv(os.path.join(raw_dir, "users.csv"), index=False)
    items_df.to_csv(os.path.join(raw_dir, "items.csv"), index=False)
    interactions_df.to_csv(os.path.join(raw_dir, "interactions.csv"), index=False)

    logging.info(f"Ingested {len(users_df)} users, {len(items_df)} items, and {len(interactions_df)} raw interactions.")
    return users_df, items_df, interactions_df

if __name__ == "__main__":
    cfg = load_config()
    generate_raw_datasets(cfg)
