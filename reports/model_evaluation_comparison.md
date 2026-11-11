# Model Evaluation Comparison Report

| Model                   |   K |   Precision@5 |   Recall@5 |   NDCG@5 |   Coverage |   Diversity |   RMSE |    MAE |   Precision@10 |   Recall@10 |   NDCG@10 |
|:------------------------|----:|--------------:|-----------:|---------:|-----------:|------------:|-------:|-------:|---------------:|------------:|----------:|
| Baseline Popularity     |   5 |        0.0042 |     0.0104 |   0.0141 |      0.028 |      0.8977 | 0.8049 | 0.6558 |       nan      |    nan      |  nan      |
| Personalized SVD Hybrid |   5 |        0.0273 |     0.0456 |   0.0741 |      0.808 |      0.5299 | 0.8016 | 0.6474 |       nan      |    nan      |  nan      |
| Baseline Popularity     |  10 |      nan      |   nan      | nan      |      0.056 |      0.855  | 0.8049 | 0.6558 |         0.0053 |      0.0203 |    0.0242 |
| Personalized SVD Hybrid |  10 |      nan      |   nan      | nan      |      0.964 |      0.5194 | 0.8016 | 0.6474 |         0.0219 |      0.077  |    0.0972 |