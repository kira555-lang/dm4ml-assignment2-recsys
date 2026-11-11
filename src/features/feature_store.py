import os
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [FEATURE_STORE] - %(levelname)s - %(message)s")

def build_feature_store(config):
    proc_dir = config["paths"]["processed_dir"]
    feat_dir = config["paths"]["features_dir"]
    os.makedirs(feat_dir, exist_ok=True)

    users_df = pd.read_parquet(os.path.join(proc_dir, "users_clean.parquet"))
    items_df = pd.read_parquet(os.path.join(proc_dir, "items_clean.parquet"))
    train_df = pd.read_parquet(os.path.join(proc_dir, "train_interactions.parquet"))
    user_sum = pd.read_parquet(os.path.join(proc_dir, "user_activity_summary.parquet"))
    item_sum = pd.read_parquet(os.path.join(proc_dir, "item_popularity_summary.parquet"))

    ref_date = train_df["timestamp"].max()

    # Calculate User Preferred Category (Genre with highest interaction frequency in train)
    train_with_cat = train_df.merge(items_df[["item_id", "category"]], on="item_id", how="left")
    pref_cat = (
        train_with_cat.groupby(["user_id", "category"])
        .size()
        .reset_index(name="cat_count")
        .sort_values(["user_id", "cat_count"], ascending=[True, False])
        .drop_duplicates("user_id")[["user_id", "category"]]
        .rename(columns={"category": "preferred_category"})
    )

    # Build Reusable User Feature Table
    user_features = users_df.merge(user_sum, on="user_id", how="left").merge(pref_cat, on="user_id", how="left")
    user_features["activity_count"] = user_features["activity_count"].fillna(0).astype(int)
    user_features["avg_rating"] = user_features["avg_rating"].fillna(train_df["rating"].mean()).round(4)
    user_features["rating_std"] = user_features["rating_std"].fillna(0.0).round(4)
    user_features["preferred_category"] = user_features["preferred_category"].fillna("Unknown")
    user_features["days_since_last_interaction"] = (
        (ref_date - pd.to_datetime(user_features["last_interaction_ts"])).dt.days.fillna(999).astype(int)
    )
    user_features = user_features.drop(columns=["last_interaction_ts"])

    # Build Reusable Item Feature Table
    max_cnt = max(item_sum["interaction_count"].max(), 1)
    item_features = item_sum.copy()
    item_features["popularity_percentile"] = item_features["interaction_count"].rank(pct=True).round(4)
    item_features["normalized_popularity"] = (item_features["interaction_count"] / max_cnt).round(4)
    item_features["item_age_years"] = (ref_date.year - item_features["release_year"]).clip(lower=0)

    # Build Enriched Interaction Feature Table for Training
    interaction_features = (
        train_df[["interaction_id", "user_id", "item_id", "rating", "timestamp"]]
        .merge(user_features[["user_id", "avg_rating", "activity_count", "preferred_category", "days_since_last_interaction"]], on="user_id", how="left")
        .merge(item_features[["item_id", "category", "item_avg_rating", "bayesian_popularity_score", "popularity_percentile"]], on="item_id", how="left")
    )
    interaction_features["category_affinity_match"] = (
        interaction_features["preferred_category"] == interaction_features["category"]
    ).astype(int)
    interaction_features["user_rating_bias"] = (
        interaction_features["avg_rating"] - train_df["rating"].mean()
    ).round(4)

    # Save all Feature Tables to Feature Store
    user_features.to_parquet(os.path.join(feat_dir, "user_features.parquet"), index=False)
    item_features.to_parquet(os.path.join(feat_dir, "item_features.parquet"), index=False)
    interaction_features.to_parquet(os.path.join(feat_dir, "interaction_features.parquet"), index=False)

    logging.info(f"Feature Store updated: {len(user_features)} user features, {len(item_features)} item features.")
    return user_features, item_features, interaction_features

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    build_feature_store(cfg)
