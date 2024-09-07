# programmers

## Training & Evaluation

```bash
$ python3 main.py --data_dir train_job --over_sampler Borderline --model_save ./model/
```

## Dataset

|        Name       | Description | Count |
| ----------------- | ----------- | ----- |
|   job_companies   | 회사-직무-규모 |  733  |
|  job_tags  | 직무-역량 | 3,477 |
|  tags | 역량-키워드 | 887 |
|  user_tags | 유저-역량 | 2,582 |
|   train  | 유저-직무-지원여부 | 6,000 |


## Performance for Test Dataset

|   Model    |       Imbalancing      | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC Score (%) | iteration |
|  ----------   | ---------------- | -------------- | ----------- | ------------------ | --------------- | ---------------- | -------- |
|  **RandomForest**   |   X   |    83.4    |     15.0    |        3.5        |      5.7       |       50.1       | ------- |
|                     |  SMOTE |      69.6      |   17.2  |      29.6      |    21.7     |     52.9     | 1 |
|                     |  SMOTE |      76.2     |   23.8  |      30.4      |    26.7     |     57.1     | 2 |


## Confusion Matrix for Test Dataset
| Matrix |   Actual O  |   Actual X   | 
|  --------- |  --------- | ---------- |
|  Predict O |   1,277  |   266   | 
|  Predict X |  196   | 61 | 
