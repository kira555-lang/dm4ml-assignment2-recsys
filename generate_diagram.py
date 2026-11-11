import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram(output_path="reports/architecture_diagram.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Title
    ax.text(
        7, 10.5,
        "End-to-End Recommendation System Data Management Pipeline",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1e293b"
    )
    ax.text(
        7, 10.1,
        "Orchestrated via Apache Airflow DAG (dags/recommendation_pipeline_dag.py) & CLI Runner",
        ha="center", va="center", fontsize=10, style="italic", color="#475569"
    )

    def draw_box(x, y, w, h, title, subtitle, facecolor="#f8fafc", edgecolor="#334155"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.68, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a")
        ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=8.5, color="#334155")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.5", color="#475569", lw=1.8)
        )
        if label:
            ax.text((x1 + x2)/2 + 0.15, (y1 + y2)/2, label, fontsize=8, color="#475569", fontweight="bold")

    # Stage 1: Ingestion
    draw_box(
        4.2, 8.6, 5.6, 1.0,
        "1. Data Ingestion (src/ingestion/ingest.py)",
        "Raw CSVs: users.csv (600), items.csv (250), interactions.csv (18,144)",
        facecolor="#e0f2fe", edgecolor="#0284c7"
    )

    # Stage 2: Validation
    draw_box(
        4.2, 7.1, 5.6, 1.0,
        "2. Data Validation (src/validation/validate.py)",
        "Checks: Nulls, Duplicates, Rating Range [1-5], Timestamps, FK Integrity",
        facecolor="#fef3c7", edgecolor="#d97706"
    )

    # Quarantine & Audit Report Side Boxes
    draw_box(
        10.5, 7.1, 3.1, 1.0,
        "Quarantine & Audit Store",
        "quarantined_interactions.csv\ndata_validation_report.json",
        facecolor="#fee2e2", edgecolor="#dc2626"
    )

    # Stage 3: Preparation
    draw_box(
        4.2, 5.6, 5.6, 1.0,
        "3. Data Preparation (src/preparation/prepare.py)",
        "Type Standardization, Deduplication & 80/20 Chronological Split",
        facecolor="#e0e7ff", edgecolor="#4f46e5"
    )

    # Stage 4: Transformation
    draw_box(
        4.2, 4.1, 5.6, 1.0,
        "4. Data Transformation (src/transformation/transform.py)",
        "User Activity Summary, Bayesian Popularity, User-Item Matrix",
        facecolor="#ede9fe", edgecolor="#7c3aed"
    )

    # Stage 5: Feature Store
    draw_box(
        4.2, 2.6, 5.6, 1.0,
        "5. Feature Management (src/features/feature_store.py)",
        "Parquet Store: user_features, item_features, interaction_features",
        facecolor="#dcfce7", edgecolor="#16a34a"
    )

    # Stage 6: Model Training
    draw_box(
        1.2, 1.1, 5.2, 1.0,
        "6A. Baseline Model (src/training)",
        "Bayesian Popularity Recommender (.pkl)",
        facecolor="#f1f5f9", edgecolor="#475569"
    )
    draw_box(
        7.6, 1.1, 5.2, 1.0,
        "6B. Personalized Model (src/training)",
        "TruncatedSVD + Genre Affinity Hybrid (.pkl)",
        facecolor="#f1f5f9", edgecolor="#475569"
    )

    # MLflow Box
    draw_box(
        0.4, 2.6, 3.1, 1.0,
        "MLflow Tracking",
        "Logs Params, Metrics & Models\n(./mlruns store)",
        facecolor="#ffedd5", edgecolor="#ea580c"
    )

    # Stage 7: Evaluation
    draw_box(
        3.5, -0.3, 7.0, 0.95,
        "7. Model Evaluation (src/evaluation/evaluate.py)",
        "Precision@K, Recall@K, NDCG@K, Coverage, Diversity, RMSE, MAE (K=5, 10)",
        facecolor="#ccfbf1", edgecolor="#0d9488"
    )

    # Connect Arrows
    draw_arrow(7.0, 8.6, 7.0, 8.1)
    draw_arrow(9.8, 7.6, 10.5, 7.6, "Invalid")
    draw_arrow(7.0, 7.1, 7.0, 6.6, "Valid")
    draw_arrow(7.0, 5.6, 7.0, 5.1)
    draw_arrow(7.0, 4.1, 7.0, 3.6)
    draw_arrow(5.8, 2.6, 3.8, 2.1)
    draw_arrow(8.2, 2.6, 10.2, 2.1)
    draw_arrow(3.8, 1.1, 5.5, 0.65)
    draw_arrow(10.2, 1.1, 8.5, 0.65)
    draw_arrow(2.5, 2.1, 2.0, 2.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()