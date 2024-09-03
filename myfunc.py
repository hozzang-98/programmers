from import_list import * 

def change_size(x):

    if x == "1000 이상":

        return 1000
    
    else:

        y = (int(x.split("-")[0]) + int(x.split("-")[1]))//2 # string 평균 값으로 대치

        return y
    
# def fill_missing_tag_size(row):

#     if pd.isnull(row['tagSize']):

#         return user_Size_dict.get(row['userID'], row['tagSize'])
    
#     return row['tagSize']


# def merge_keyword(x, sorting_col):

#     y = pd.merge(x, tags, how = 'inner', left_on = 'tagID', right_on = 'tagID')
#     y.sort_values(by=[sorting_col,'keyword'], inplace=True)
    
#     return y

# def concat_keywords(tagkeyword_df, sorting_col):

#     y = tagkeyword_df.groupby(sorting_col).apply(lambda x: ', '.join(x.keyword)).reset_index().rename(columns={0:'{}_keywords'.format(sorting_col[:-2])})

#     return y

# def merge_company_size(x):

#     y = pd.merge(x, job_companies, how='inner', left_on = 'jobID', right_on = 'jobID').drop(columns='companyID')[['userID','jobID','companySize','applied']]

#     return y

# def merge_user_f_size(x):

#     y = pd.merge(x, user_avgSize, how = 'inner', left_on = 'userID', right_on = 'userID').rename(columns={'avg_of_size':'userSize'})
#     y['userSize'].fillna(y['userSize'].mean(), inplace=True)
#     return y

# def merge_tagkeyword(x):

#     y = pd.merge(x, job_tagkeyword_concat, how='inner', left_on = 'jobID', right_on = 'jobID').rename(columns={'keywords':'job_keywords'})[['userID','userSize','jobID','companySize','job_keywords','applied']]

#     return y

# def merge_tagkeyword_concat(x):

#     y = pd.merge(x, user_tagkeyword_concat, how='inner', left_on = 'userID', right_on = 'userID')[['userID','userSize','user_keywords','jobID','companySize','job_keywords','applied']]

#     return y

def calculate_cosine_similarity(row):

    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform([row['user_keywords'], row['job_keywords']])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    except: # 길이기 다른 경우 계산이 안되는 문제 확인
        
        cosine_sim = -9999

    return cosine_sim

def calculate_jaccard_similarity(row):

    separation = ', '

    set1 = set(row['user_keywords'].split(separation))
    set2 = set(row['job_keywords'].split(separation))

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)

def calculate_std_sizediff(x):

    x['size_diff'] = x['userSize'] - x['companySize']
    # scale_list = ['size_diff']
    scale_list = ['userSize','companySize','size_diff']
    scaler = MinMaxScaler()

    x[scale_list] = scaler.fit_transform(x[scale_list])

    return x 

# def scaliing(x, column_list):
    
#     scaler = MinMaxScaler()

#     x[scale_list] = scaler.fit_transform(x[column_list])

#     return x 

def KFold_RandomForest(x,y):

    kfold = StratifiedKFold(n_splits=5).split(x, y)

    scores = []
    model = RandomForestClassifier(n_estimators=500, random_state=2022)
    for k, (tri, val) in enumerate(kfold):
        
        model.fit(x.iloc[tri], y.iloc[tri])

        score = model.score(x.iloc[val], y.iloc[val])
        scores.append(score)
        print(f'fold {k+1} 모델의 정확도: {score*100:.2f}%')

    print(f"CV 정확도: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    return model

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