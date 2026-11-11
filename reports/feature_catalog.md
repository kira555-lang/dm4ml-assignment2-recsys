## Feature Catalog
| Feature Name | Entity Level | Data Type | Source Table | Description & Logic |
| :--- | :--- | :--- | :--- | :--- |
| `activity_count` | User | `int64` | `user_features.parquet` | Total historical validated interactions per user. |
| `avg_rating` | User | `float64` | `user_features.parquet` | User's mean rating across training history. |
| `rating_std` | User | `float64` | `user_features.parquet` | Standard deviation of user's historical ratings. |
| `preferred_category` | User | `string` | `user_features.parquet` | Most frequently interacted movie genre for the user. |
| `days_since_last_interaction` | User | `int64` | `user_features.parquet` | Days elapsed since the user's last recorded interaction. |
| `interaction_count` | Item | `int64` | `item_features.parquet` | Total number of unique training users interacting with item. |
| `item_avg_rating` | Item | `float64` | `item_features.parquet` | Mean explicit rating received by the item. |
| `bayesian_popularity_score` | Item | `float64` | `item_features.parquet` | Bayesian damped rating (`m=10`) preventing small-sample bias. |
| `popularity_percentile` | Item | `float64` | `item_features.parquet` | Percentile rank (`0.0 - 1.0`) of item interaction volume. |
| `item_age_years` | Item | `int64` | `item_features.parquet` | Years elapsed since the item's `release_year`. |
| `category_affinity_match` | Interaction | `int64` | `interaction_features.parquet` | `1` if item genre matches user's `preferred_category`, else `0`. |
| `user_rating_bias` | Interaction | `float64` | `interaction_features.parquet` | User's rating offset relative to global mean rating. |
