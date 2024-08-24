# programmers


## Dataset

|        Name       | Description | Count |
| ----------------- | ----------- | ----- |
|   job_companies   | 회사-직무-규모 |  733  |
|  job_tags  | 직무-역량 | 3,477 |
|  tags | 역량-키워드 | 887 |
|  user_tags | 유저-역량 | 2,582 |
|   train  | 유저-직무-지원여부 | 6,000 |


## Results for Test Dataset

|   Model    |       Imbalancing      | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC Score (%) |
|  ----------   | ---------------- | -------------- | ----------- | ------------------ | --------------- | ---------------- |
|  **RandomForest**   |   X   |    **83.4**    |     15.0    |        3.5        |      5.7       |       50.1       |
|                     |  SMOTE |      69.6      |   **17.2**  |      **29.6**      |    **21.7**     |     **52.9**     |
