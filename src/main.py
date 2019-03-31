import pandas as pd
import pandas_profiling as pdp
import multiprocessing
import setting
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pickle
import os
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime

DATA_DIR_PATH = '../data/'
PROFILE_DIR_PATH = '../profile/'
SUBMIT_DIR_PATH = '../submit/'
MODEL_DIR_PATH = '../model/'
ID = 'ID_code'
TARGET = 'target'
dtypes = setting.DTYPES
NUM_ROUND = 1000
N_FOLD = 10


# return path exist (true or false)
def check_filepath(filepath):
    return os.path.isfile(filepath)


# dataset: (train, test)
def load_file(dataset):
    usecols = dtypes.keys()
    if dataset == 'test':
        usecols = [col for col in dtypes.keys() if col != TARGET]
    df = pd.read_csv(DATA_DIR_PATH + f'{dataset}.csv', encoding='utf-8', dtype=dtypes, usecols=usecols)
    return df


# profiling report to html file
def output_profile(df, filename):
    filepath = PROFILE_DIR_PATH + filename
    if check_filepath(filepath):
        print(f'already exist file path <{filepath}>')
    else:
        print(f'create profiling report <{filepath}>')
        profile = pdp.ProfileReport(df)
        profile.to_file(filepath)


# smote oversampling train data
def smote_sampling(X, y):
    print(X.shape)
    print(y.shape)
    ds_values = y.value_counts()
    print(ds_values)
    # max_index = ds_values.idxmax()
    smote = SMOTE(sampling_strategy='auto', random_state=0, n_jobs=-1)
    X_resample, y_resample = smote.fit_resample(X, y)
    return X_resample, y_resample


def my_oversampling(data):
    print(data.shape)
    print(data[TARGET].value_counts())
    positive_data = data[data[TARGET] == '1']
    negative_data = data[data[TARGET] == '0']
    data = pd.concat([data, positive_data.sample(frac=1)], axis=0)
    data = pd.concat([data, positive_data.sample(frac=1)], axis=0)
    data = pd.concat([data, negative_data.sample(frac=1)], axis=0)
    print(data.shape)
    return data


# scaling data (numpy)
def scaling(data):
    data = StandardScaler().fit_transform(data)
    return data


def build_xgb_model(xgtrain):
    xgb_clf = xgb.XGBClassifier(n_estimators=200,
                                learning_rate=0.1,
                                max_depth=5,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                scale_pos_weight=1,
                                objective='binary:logistic',
                                # nthread=8,
                                # silent=False
                                )
    xgb_param = xgb_clf.get_xgb_params()
    print('Start cross validation')
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=5, metrics=['auc'],
                      early_stopping_rounds=10, stratified=True, seed=0)
    print('Best number of trees = {}'.format(cvresult.shape[0]))
    xgb_clf.set_params(n_estimators=cvresult.shape[0])
    return xgb_clf, cvresult.shape[0]


def build_lgb_model(train, test):
    folds = StratifiedKFold(n_splits=N_FOLD, shuffle=False, random_state=0)
    preds = np.zeros(test.shape[0])
    oof = np.zeros(train.shape[0])

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[TARGET])):
        print('fold: ', fold_idx)
        lgb_train = lgb.Dataset(train.iloc[train_idx][features], label=train.iloc[train_idx][TARGET])
        lgb_valid = lgb.Dataset(train.iloc[valid_idx][features], label=train.iloc[valid_idx][TARGET], reference=lgb_train)
        params = {
            'task': 'train',
            'objective': 'binary',
            'bagging_freq': 5,
            'bagging_fraction': 0.4,
            'boost_from_average': 'False',
            'boost': 'gbdt',
            'feature_fraction': 0.05,
            'learning_rate': 0.01,
            'max_depth': -1,
            'metric': 'auc',
            'min_data_in_leaf': 80,
            'min_sum_hessian_in_leaf': 10.0,
            'num_leaves': 13,
            'num_threads': 8,
            'tree_learner': 'serial',
            'verbosity': 1,
            'is_unbalance': True
        }
        clf = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        valid_names=['tarin', 'valid'],
                        num_boost_round=NUM_ROUND,
                        verbose_eval=100,
                        early_stopping_rounds=50)
        oof[valid_idx] = clf.predict(train.iloc[valid_idx][features], num_iteration=clf.best_iteration)
        preds += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
    oof_score = roc_auc_score(train[TARGET], oof)
    print(oof)
    # acc_score, class_report, auc_score = model_score(train[TARGET], oof)
    return preds, oof_score


# return two score
def model_score(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    print('accuracy score: ' + str(acc_score))
    print('auc score', auc_score)
    print('classification report', class_report)

    return acc_score, class_report, auc_score


if __name__ == '__main__':
    #
    # input data to dataframe
    #

    with multiprocessing.Pool() as pool:
        df_train, df_test = pool.map(load_file, ["train", "test"])

    # df_train = df_train.sample(5000)
    # df_test = df_test.sample(5000)
    print(df_train.info())
    print(df_test.info())

    output_profile(df_train, 'train.html')
    output_profile(df_test, 'test.html')
    output_profile(df_train[df_train[TARGET] == 0], 'train_0.html')
    output_profile(df_train[df_train[TARGET] == 1], 'train_1.html')

    features = [c for c in df_train.columns if c not in [ID, TARGET]]

    #
    # pre processing (oversampling and scaling)
    #

    df_train = my_oversampling(df_train)
    # df_train[features], df_train[TARGET] = smote_sampling(df_train[features], df_train[TARGET])

    # df_train[features] = scaling(df_train[features])
    # df_test[features] = scaling(df_test[features])

    #
    # training model and save
    #

    y_pred, total_score = build_lgb_model(df_train, df_test)

    #
    # print predict and score
    #

    # y_valid_pred = clf.predict(x_valid, num_iteration=clf.best_iteration)
    # print(y_valid_pred)
    # acc_score, class_report, auc_score = model_score(y_valid_pred, y_valid)
    # print(lgb_clf.best_score['valid_0']['auc'])

    #
    # print importance features
    #

    # df_feat_imp = pd.Series(lgb_clf.feature_importance(), index=features)
    # print(df_feat_imp.sort_values(ascending=False))

    #
    # test to predict and submit dataframe
    #

    # y_proba = lgb_clf.predict(x_test)
    # y_pred = np.where(y_pred > 0.5, 1, 0)
    # y_pred = np.argmax(y_pred, axis=1)
    df_test_pred = pd.Series(y_pred, index=df_test[ID], name=TARGET)
    str_nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
    df_test_pred.to_csv(SUBMIT_DIR_PATH + f'submit_{str_nowtime}_{round(total_score * 100, 2)}.csv',
                        header=True)
