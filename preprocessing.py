from baisic_preprocessing import *
from sklearn.model_selection import train_test_split

def preparing_Merge(data1, data2, data3, data4):

    data4.drop_duplicates(subset=['userID','tagID'], inplace=True) # 유저당 직무 태그 중복 존재하므로 제거 필요

    data1['companySize'] = data1['companySize'].apply(lambda x:change_size(x) if type(x) == str else x)

    avg_of_size = int(data1['companySize'].mean())

    data1['companySize'].fillna(avg_of_size, inplace=True)

    tag_size = pd.merge(data2, data1[['jobID','companySize']], how='inner', left_on = 'jobID', right_on = 'jobID')

    tag_avg_size = tag_size.groupby('tagID')['companySize'].mean().reset_index().rename(columns={'companySize':'tagSize'})

    user_tag_size = pd.merge(data4, tag_avg_size, how = 'left', left_on = 'tagID', right_on = 'tagID')

    user_Size_df = user_tag_size.groupby('userID')['tagSize'].mean().reset_index()

    user_Size_dict = user_Size_df.set_index('userID')['tagSize'].to_dict()

    user_tag_size['tagSize'] = user_tag_size.apply(lambda x: fill_missing_tag_size(user_Size_dict, x), axis=1)
    user_tag_size['tagSize'].fillna(user_tag_size['tagSize'].mean(),inplace=True)

    user_avgSize = user_tag_size.groupby('userID')['tagSize'].mean().reset_index().rename(columns={'tagSize':'avg_of_size'})

    job_tagkeyword = merge_keyword(data2, data3, 'jobID')
    user_tagkeyword = merge_keyword(data4, data3, 'userID')

    job_tagkeyword_concat = concat_keywords(job_tagkeyword, 'jobID')
    user_tagkeyword_concat = concat_keywords(user_tagkeyword, 'userID')

    return data1, user_avgSize, job_tagkeyword_concat, user_tagkeyword_concat

def do_merge(job_companies, user_avgSize, job_tagkeyword_concat, user_tagkeyword_concat, df):

    # train_data, test_data = train_test_split(train, train_size=0.7, stratify=train['applied'])

    df1 = merge_company_size(df, job_companies)
    # test_data1 = merge_company_size(test_data, job_companies)

    df2 = merge_user_f_size(df1, user_avgSize)
    # test_data2 = merge_user_f_size(test_data1, user_avgSize)

    df3 = merge_tagkeyword(df2, job_tagkeyword_concat)
    # test_data3 = merge_tagkeyword(test_data2, job_tagkeyword_concat)

    df4 = merge_tagkeyword_concat(df3, user_tagkeyword_concat)
    # test_data4 = merge_tagkeyword_concat(test_data3, user_tagkeyword_concat)

    df4['keywords_similarity'] = df4.apply(calculate_jaccard_similarity, axis=1)
    # test_data4['keywords_similarity'] = test_data4.apply(calculate_jaccard_similarity, axis=1)

    df5 = calculate_std_sizediff(df4)
    # test_data5 = calculate_std_sizediff(test_data4)

    df5 = df5[['userID','userSize','user_keywords','jobID','companySize','job_keywords','keywords_similarity','size_diff','applied']]
    # test_data5 = test_data5[['userID','userSize','user_keywords','jobID','companySize','job_keywords','keywords_similarity','size_diff','applied']]

    X = df5[['userSize','companySize','keywords_similarity']]
    y = df5[['applied']]

    # X_test = test_data5[['userSize','companySize','keywords_similarity']]
    # y_test = test_data5[['applied']]

    return X, y