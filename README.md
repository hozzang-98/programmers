# programmers


## Dataset

|        Name       | Description | Count |
| ----------------- | ----------- | ----- |
|   job_companies   | 회사-직무-규모 |  733  |
|  job_tags  | 직무-역량 | 3,477 |
|  tags | 역량-키워드 | 887 |
|  user_tags | 유저-역량 | 2,582 |
|   train  | 유저-직무-지원여부 | 6,000 |

Accuracy: 0.70
Precision: 0.17
Recall: 0.30
F1 Score: 0.22
Confusion Matrix:
[[1176  367]
 [ 181   76]]
ROC-AUC Score: 0.53

## Results

|   Model    |       Imbalancing      | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC Score (%) |
|  ----------   | ---------------- | -------------- | ----------- | ------------------ | --------------- | ---------------- |
|  **RandomForest**   |   X   |    **89.3**    |     78.4    |        74.5        |      81.9       |       60.4       |
|               |  SMOTE |      89.1      |   **80.6**  |      **77.4**      |    **84.2**     |     **62.3**     |
