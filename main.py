import warnings
warnings.filterwarnings('ignore')

import joblib
from data_loader import load_data
from preprocessing import preparing_Merge, do_merge
from sklearn.model_selection import train_test_split
from modeling import over_sampling, KFold_RandomForest, randomforest, performace_check

import argparse

def main(args):

    # 1. Data Load
    # data_dir = "train_job"
    job_companies, job_tags, tags, user_tags, origin_train = load_data(args)

    # 2. Preprocessing1
    job_companies, user_avgSize, job_tagkeyword_concat, user_tagkeyword_concat = preparing_Merge(job_companies, job_tags, tags, user_tags)

    # 3. Split
    train_data, test_data = train_test_split(origin_train, train_size=0.7, stratify=origin_train['applied'])

    # 3. Preprocessing2
    X_train, y_train = do_merge(job_companies, user_avgSize, job_tagkeyword_concat, user_tagkeyword_concat, train_data)
    X_test, y_test = do_merge(job_companies, user_avgSize, job_tagkeyword_concat, user_tagkeyword_concat, test_data)

    # 4. Imbalancing
    X_train_resampled, y_train_resampled = over_sampling(X_train, y_train, args)

    # 5. Modeling
    model_name = "RF"
    KFold_RandomForest(X_train_resampled, y_train_resampled, args, model_name)
    model = joblib.load(args.model_save + model_name)

    y_pred = model.predict(X_test)

    performace_check(y_test, y_pred)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./train_job/", type=str, help="input data dir")
    parser.add_argument("--over_sampler", default="SMOTE", type=str, help="over sampling method")
    parser.add_argument("--model_save", default="./model/", type=str, help="model save dir")
    args = parser.parse_args()

    main(args)