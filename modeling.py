import numpy as np, joblib
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE

def over_sampling(x, y, args):

    MODEL_CLASSES = {
    'SMOTE': SMOTE,
    'Borderline' : BorderlineSMOTE,
    'KMeans': KMeansSMOTE
                    }
    print(f"{MODEL_CLASSES[args.over_sampler]} 기법을 활용해 Over Sampling을 진행합니다.")
    Over_Sampler = MODEL_CLASSES[args.over_sampler]()

    X_resampled, y_resampled = Over_Sampler.fit_resample(x, y)

    return X_resampled, y_resampled

def KFold_RandomForest(x, y, args, model_name):

    kfold = StratifiedKFold(n_splits=5).split(x, y)

    scores = []

    
    model = RandomForestClassifier(n_estimators=500, random_state=2022)

    for k, (tri, val) in enumerate(kfold):
        
        model.fit(x.iloc[tri], y.iloc[tri])

        score = model.score(x.iloc[val], y.iloc[val])
        scores.append(score)
        print(f'fold {k+1} 모델의 정확도: {score*100:.2f}%')

    print(f"CV 정확도: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    joblib.dump(model, args.model_save + model_name)

def randomforest(x, y, model_path): # KFold + GridSearchCV

    model_name = "RF"
    classifier = RandomForestClassifier(random_state=1210)
    param_grid = {
                    'n_estimators': [100, 200],       # 트리 개수
                    'max_depth': [20, 30],      # 최대 깊이
                    'min_samples_split': [2, 5, 10],      # 분할을 위한 최소 샘플 수
                    'min_samples_leaf': [1, 2, 4],        # 리프 노드의 최소 샘플 수
                    'max_features': ['auto', 'sqrt', 'log2'],  # 각 트리에서 사용할 최대 특성 수
                    'bootstrap': [True, False],           # 부트스트랩 샘플링 여부
                    'criterion': ['gini', 'entropy']      # 분할 기준
                }
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1210)

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                            cv=kfold, verbose=0, n_jobs=-1) 
    grid_search.fit(x, y)

    # 최적의 하이퍼파라미터 출력
    print(f"Best parameters found: {grid_search.best_params_}")

    # 테스트 세트에 대해 성능 평가
    best_rf = grid_search.best_estimator_
    
    joblib.dump(best_rf, model_path + model_name)
    print("모델 저장 완료")
    
def performace_check(real, predict):

    # 정확도 (Accuracy)
    accuracy = accuracy_score(real, predict)
    print(f"Accuracy: {accuracy:.3f}")

    # 정밀도 (Precision)
    precision = precision_score(real, predict)
    print(f"Precision: {precision:.3f}")

    # 재현율 (Recall)
    recall = recall_score(real, predict)
    print(f"Recall: {recall:.3f}")

    # F1 점수 (F1 Score)
    f1 = f1_score(real, predict)
    print(f"F1 Score: {f1:.3f}")

    # 혼동 행렬 (Confusion Matrix)
    cm = confusion_matrix(real, predict)
    print("Confusion Matrix:")
    print(cm)

    # ROC-AUC 점수 (ROC-AUC Score)
    roc_auc = roc_auc_score(real, predict)
    print(f"ROC-AUC Score: {roc_auc:.3f}")