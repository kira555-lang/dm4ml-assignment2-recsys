import os
import json
import yaml
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [VALIDATION] - %(levelname)s - %(message)s")

def run_validation(config):
    raw_dir = config["paths"]["raw_dir"]
    errors_dir = config["paths"]["errors_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    users_df = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    items_df = pd.read_csv(os.path.join(raw_dir, "items.csv"))
    inter_df = pd.read_csv(os.path.join(raw_dir, "interactions.csv"))

    total_raw = len(inter_df)
    error_records = []

    # Check 1: Missing Values
    missing_mask = inter_df[["user_id", "item_id", "rating", "timestamp"]].isnull().any(axis=1)
    if missing_mask.sum() > 0:
        err = inter_df[missing_mask].copy()
        err["error_reason"] = "MISSING_CRITICAL_FIELD"
        error_records.append(err)

    # Check 2: Exact Duplicates (on user_id, item_id, timestamp)
    dup_mask = inter_df.duplicated(subset=["interaction_id", "user_id", "item_id", "timestamp"], keep="first")
    if dup_mask.sum() > 0:
        err = inter_df[dup_mask].copy()
        err["error_reason"] = "DUPLICATE_RECORD"
        error_records.append(err)

    # Check 3: Data Type & Rating Range [1.0, 5.0]
    numeric_rating = pd.to_numeric(inter_df["rating"], errors="coerce")
    invalid_rating_mask = numeric_rating.isnull() | (numeric_rating < 1.0) | (numeric_rating > 5.0)
    if invalid_rating_mask.sum() > 0:
        err = inter_df[invalid_rating_mask].copy()
        err["error_reason"] = "INVALID_RATING_RANGE_OR_TYPE"
        error_records.append(err)

    # Check 4: Timestamp Validity & Future Date Check
    parsed_ts = pd.to_datetime(inter_df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    now_ts = pd.Timestamp.now()
    invalid_ts_mask = parsed_ts.isnull() | (parsed_ts > now_ts) | (parsed_ts < pd.Timestamp("2000-01-01"))
    if invalid_ts_mask.sum() > 0:
        err = inter_df[invalid_ts_mask].copy()
        err["error_reason"] = "INVALID_OR_FUTURE_TIMESTAMP"
        error_records.append(err)

    # Check 5: Referential Integrity (Foreign Key check against Users & Items)
    valid_users = set(users_df["user_id"].dropna().unique())
    valid_items = set(items_df["item_id"].dropna().unique())
    orphan_user_mask = ~inter_df["user_id"].isin(valid_users) & ~inter_df["user_id"].isnull()
    orphan_item_mask = ~inter_df["item_id"].isin(valid_items) & ~inter_df["item_id"].isnull()
    ref_integrity_mask = orphan_user_mask | orphan_item_mask
    if ref_integrity_mask.sum() > 0:
        err = inter_df[ref_integrity_mask].copy()
        err["error_reason"] = "REFERENTIAL_INTEGRITY_VIOLATION"
        error_records.append(err)

    # Combine all invalid masks
    combined_invalid_mask = (
        missing_mask | dup_mask | invalid_rating_mask | invalid_ts_mask | ref_integrity_mask
    )

    valid_inter_df = inter_df[~combined_invalid_mask].copy()
    if error_records:
        quarantine_df = pd.concat(error_records, ignore_index=True).drop_duplicates(subset=["interaction_id", "error_reason"])
    else:
        quarantine_df = pd.DataFrame(columns=list(inter_df.columns) + ["error_reason"])

    # Save quarantined records
    quarantine_path = os.path.join(errors_dir, "quarantined_interactions.csv")
    quarantine_df.to_csv(quarantine_path, index=False)

    # Save validated interactions for next stage
    valid_inter_df.to_csv(os.path.join(raw_dir, "validated_interactions.csv"), index=False)

    # Generate Formal Validation Report
    report = {
        "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_raw_interactions": int(total_raw),
        "total_valid_interactions": int(len(valid_inter_df)),
        "total_quarantined_records": int(combined_invalid_mask.sum()),
        "pass_rate_percentage": round(len(valid_inter_df) / total_raw * 100, 2),
        "Check_Breakdown": {
            "missing_values_count": int(missing_mask.sum()),
            "duplicate_records_count": int(dup_mask.sum()),
            "invalid_ratings_count": int(invalid_rating_mask.sum()),
            "invalid_timestamps_count": int(invalid_ts_mask.sum()),
            "referential_integrity_failures": int(ref_integrity_mask.sum())
        },
        "quarantine_file": quarantine_path
    }

    report_path = os.path.join(reports_dir, "data_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logging.info(f"Validation complete: {len(valid_inter_df)}/{total_raw} passed ({report['pass_rate_percentage']}%). Quarantined {combined_invalid_mask.sum()} records.")
    return report

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_validation(cfg)
