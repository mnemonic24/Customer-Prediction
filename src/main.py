import pandas as pd
import pandas_profiling as pdp
import multiprocessing
import setting
import metric
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pickle
import os
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime

DATA_DIR_PATH = '../data/'
PROFILE_DIR_PATH = '../profile/'
SUBMIT_DIR_PATH = '../submit/'
MODEL_DIR_PATH = '../model/'
ID = 'ID_code'
TARGET = 'target'
dtypes = setting.DTYPES
NUM_ROUND = 10000
N_FOLD = 5


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
def smote_sampling(df):
    print(df.shape)
    print(df[TARGET].value_counts())
    # max_index = ds_values.idxmax()
    smote = SMOTE(sampling_strategy='auto', random_state=0, n_jobs=-1)
    np_x, np_y = smote.fit_sample(df[features], df[TARGET])
    print(np_x.shape)
    print(np_y.shape)
    df_x = pd.DataFrame(np_x, columns=features)
    df_y = pd.Series(np_y, name=TARGET)
    df = pd.concat([df_x, df_y], axis=1)
    print(df.shape)
    return df


# oversampling (label_0 * 2, label_1 * 3)
def my_oversampling(df):
    print(df.shape)
    print(df[TARGET].value_counts())
    positive_data = df[df[TARGET] == 1]
    negative_data = df[df[TARGET] == 0]
    df = pd.concat([df, positive_data.sample(frac=1)], axis=0)
    df = pd.concat([df, positive_data.sample(frac=1)], axis=0)
    df = pd.concat([df, negative_data.sample(frac=1)], axis=0)
    print(df.shape)
    return df


# scaling data (numpy)
def scaling(data):
    data = StandardScaler().fit_transform(data)
    return data


# LightGBM and K-Fold
# training model and plot predictions
def build_lgb_model(train, test, valid):
    folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=0)
    valid_pred = np.zeros(valid.shape[0])
    predictions = np.zeros(test.shape[0])
    oof = np.zeros(train.shape[0])

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[TARGET])):
        print('fold: ', fold_idx)
        lgb_train = lgb.Dataset(train.iloc[train_idx][features], label=train.iloc[train_idx][TARGET])
        lgb_valid = lgb.Dataset(train.iloc[valid_idx][features], label=train.iloc[valid_idx][TARGET], reference=lgb_train)
        params = {
            'bagging_freq': 5,
            'bagging_fraction': 0.335,
            'boost_from_average': 'false',
            'boost': 'gbdt',
            'feature_fraction': 0.041,
            'learning_rate': 0.0083,
            'max_depth': -1,
            'metric':'auc',
            'min_data_in_leaf': 80,
            'min_sum_hessian_in_leaf': 10.0,
            'num_leaves': 13,
            'num_threads': 8,
            'tree_learner': 'serial',
            'objective': 'binary',
            'verbosity': -1
        }
        clf = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        valid_names=['tarin', 'valid'],
                        num_boost_round=NUM_ROUND,
                        verbose_eval=1000,
                        early_stopping_rounds=1000)
        oof[valid_idx] = clf.predict(train.iloc[valid_idx][features], num_iteration=clf.best_iteration)
        predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
        valid_pred += clf.predict(valid[features], num_iteration=clf.best_iteration) / folds.n_splits
    oof_score = roc_auc_score(train[TARGET], oof)
    print("Valid Score: ", roc_auc_score(valid[TARGET], valid_pred))
    return predictions, oof_score


# Keras
def build_nn_model(train, test, valid):
    folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=0)
    predictions = np.zeros(test.shape[0])
    oof = np.zeros(train.shape[0])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
    model = Sequential()
    model.add(Dense(1000, input_shape=(200, )))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metric.auc_roc])

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[TARGET])):
        print('fold: ', fold_idx)
        # train, valid = train_test_split(train, test_size=0.1)

        model.fit(train.iloc[train_idx][features], train.iloc[train_idx][TARGET],
                  epochs=5,
                  batch_size=32,
                  validation_data=(train.iloc[valid_idx][features], train.iloc[valid_idx][TARGET]),
                  verbose=1,
                  class_weight='balanced',
                  callbacks=[early_stopping])
        oof[valid_idx] = (model.predict(train.iloc[valid_idx][features], batch_size=64)).T[0]
        predictions += (model.predict(test[features], batch_size=32) / folds.n_splits).T[0]
        print(predictions)
    oof_score = roc_auc_score(train[TARGET], oof)

    # because predict is (n, 1) shape
    return predictions, oof_score


# return scores (Accuracy, ClassificationReport, AUC)
def model_score(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    print('accuracy score: ' + str(acc_score))
    print('auc score', auc_score)
    print('classification report', class_report)

    return acc_score, class_report, auc_score


def output_submit_csv(pred, score, index, filename):
    df_test_pred = pd.Series(pred, index=index, name=TARGET)
    str_nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
    df_test_pred.to_csv(SUBMIT_DIR_PATH + f'{filename}_{str_nowtime}_{round(score * 100, 2)}.csv', header=True)


if __name__ == '__main__':
    #
    # input data to dataframe
    #

    with multiprocessing.Pool() as pool:
        df_train, df_test = pool.map(load_file, ["train", "test"])

    # df_train = df_train.sample(frac=0.1)
    # df_test = df_test.sample(frac=0.1)
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
    df_train = smote_sampling(df_train)

    df_train[features] = scaling(df_train[features])
    df_test[features] = scaling(df_test[features])

    output_profile(df_train, 'train_preprocessing.html')
    output_profile(df_test, 'test_preprocessing.html')

    df_train, df_valid = train_test_split(df_train, test_size=0.4, random_state=0, shuffle=True)

    #
    # training model and save
    #

    lgb_pred, lgb_score = build_lgb_model(df_train, df_test, df_valid)
    print(lgb_pred)
    # nn_pred, nn_score = build_nn_model(df_train, df_test, df_valid)
    # print(nn_pred)

    #
    # merge predict data and score
    #

    # ensemble_pred = (lgb_pred + nn_pred) / 2
    # ensemble_score = (lgb_score + nn_score) / 2

    #
    # test to predict and submit dataframe
    #

    output_submit_csv(lgb_pred, lgb_score, index=df_test[ID], filename='lgb')
    # output_submit_csv(nn_pred, nn_score, index=df_test[ID], filename='nn')
    # output_submit_csv(ensemble_pred, ensemble_score, index=df_test[ID], filename='ensemble')
