from myfunc import *

def fill_missing_tag_size(row):

    if pd.isnull(row['tagSize']):

        return user_Size_dict.get(row['userID'], row['tagSize'])
    
    return row['tagSize']


def merge_keyword(x, sorting_col):

    y = pd.merge(x, tags, how = 'inner', left_on = 'tagID', right_on = 'tagID')
    y.sort_values(by=[sorting_col,'keyword'], inplace=True)
    
    return y

def concat_keywords(tagkeyword_df, sorting_col):

    y = tagkeyword_df.groupby(sorting_col).apply(lambda x: ', '.join(x.keyword)).reset_index().rename(columns={0:'{}_keywords'.format(sorting_col[:-2])})

    return y

def merge_company_size(x):

    y = pd.merge(x, job_companies, how='inner', left_on = 'jobID', right_on = 'jobID').drop(columns='companyID')[['userID','jobID','companySize','applied']]

    return y

def merge_user_f_size(x):

    y = pd.merge(x, user_avgSize, how = 'inner', left_on = 'userID', right_on = 'userID').rename(columns={'avg_of_size':'userSize'})
    y['userSize'].fillna(y['userSize'].mean(), inplace=True)
    return y

def merge_tagkeyword(x):

    y = pd.merge(x, job_tagkeyword_concat, how='inner', left_on = 'jobID', right_on = 'jobID').rename(columns={'keywords':'job_keywords'})[['userID','userSize','jobID','companySize','job_keywords','applied']]

    return y

def merge_tagkeyword_concat(x):

    y = pd.merge(x, user_tagkeyword_concat, how='inner', left_on = 'userID', right_on = 'userID')[['userID','userSize','user_keywords','jobID','companySize','job_keywords','applied']]

    return y

# 1. Data Load

# 회사별 직무, 규모
job_companies = pd.read_csv('train_job/job_companies.csv')

# 직무별 태그
job_tags = pd.read_csv('train_job/job_tags.csv')

# 태그별 키워드
tags = pd.read_csv('train_job/tags.csv')

# 유저별 관심 기술
user_tags = pd.read_csv('train_job/user_tags.csv')

# 유저별 직무, 지원 여부
train = pd.read_csv('train_job/train.csv')

# 2. Preprocessing
user_tags.drop_duplicates(subset=['userID','tagID'], inplace=True) # 유저당 직무 태그 중복 존재하므로 제거 필요

job_companies['companySize'] = job_companies['companySize'].apply(lambda x:change_size(x) if type(x) == str else x)

avg_of_size = int(job_companies['companySize'].mean())

job_companies['companySize'].fillna(avg_of_size, inplace=True)

tag_size = pd.merge(job_tags, job_companies[['jobID','companySize']], how='inner', left_on = 'jobID', right_on = 'jobID')

tag_avg_size = tag_size.groupby('tagID')['companySize'].mean().reset_index().rename(columns={'companySize':'tagSize'})

user_tag_size = pd.merge(user_tags, tag_avg_size, how = 'left', left_on = 'tagID', right_on = 'tagID')

user_Size_df = user_tag_size.groupby('userID')['tagSize'].mean().reset_index()

user_Size_dict = user_Size_df.set_index('userID')['tagSize'].to_dict()

user_tag_size['tagSize'] = user_tag_size.apply(fill_missing_tag_size, axis=1)

user_avgSize = user_tag_size.groupby('userID')['tagSize'].mean().reset_index().rename(columns={'tagSize':'avg_of_size'})

job_tagkeyword = merge_keyword(job_tags, 'jobID')
user_tagkeyword = merge_keyword(user_tags, 'userID')

job_tagkeyword_concat = concat_keywords(job_tagkeyword, 'jobID')
user_tagkeyword_concat = concat_keywords(user_tagkeyword, 'userID')

# 3. modeling
train_data, test_data = train_test_split(train, train_size=0.7, stratify=train['applied'])

train_data1 = merge_company_size(train_data)
test_data1 = merge_company_size(test_data)

train_data2 = merge_user_f_size(train_data1)
test_data2 = merge_user_f_size(test_data1)

train_data3 = merge_tagkeyword(train_data2)
test_data3 = merge_tagkeyword(test_data2)

train_data4 = merge_tagkeyword_concat(train_data3)
test_data4 = merge_tagkeyword_concat(test_data3)

train_data4['keywords_similarity'] = train_data4.apply(calculate_jaccard_similarity, axis=1)
test_data4['keywords_similarity'] = test_data4.apply(calculate_jaccard_similarity, axis=1)

train_data5 = calculate_std_sizediff(train_data4)
test_data5 = calculate_std_sizediff(test_data4)

train_data5 = train_data5[['userID','userSize','user_keywords','jobID','companySize','job_keywords','keywords_similarity','size_diff','applied']]
test_data5 = test_data5[['userID','userSize','user_keywords','jobID','companySize','job_keywords','keywords_similarity','size_diff','applied']]

X_train = train_data5[['userSize','companySize','keywords_similarity']]
y_train = train_data5[['applied']]

smote = BorderlineSMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

forest = KFold_RandomForest(X_train_resampled, y_train_resampled)

X_test = test_data4[['userSize','companySize','keywords_similarity']]
y_test = test_data4[['applied']]

y_pred = forest.predict(X_test)

performace_check(y_test, y_pred)