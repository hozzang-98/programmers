import pandas as pd

def load_data(args):

    # 회사별 직무, 규모
    job_companies = pd.read_csv('{}/job_companies.csv'.format(args.data_dir))

    # 직무별 태그
    job_tags = pd.read_csv('{}/job_tags.csv'.format(args.data_dir))

    # 태그별 키워드
    tags = pd.read_csv('{}/tags.csv'.format(args.data_dir))

    # 유저별 관심 기술
    user_tags = pd.read_csv('{}/user_tags.csv'.format(args.data_dir))

    # 유저별 직무, 지원 여부
    train = pd.read_csv('{}/train.csv'.format(args.data_dir))

    return job_companies, job_tags, tags, user_tags, train
