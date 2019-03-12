import pandas as pd
import pandas_profiling as pdp
import multiprocessing
import setting
import xgboost as xgb
import pickle
import os
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

DATA_DIR_PATH = '../data/'
PROFILE_DIR_PATH = '../profile/'
SUBMIT_DIR_PATH = '../submit/'
MODEL_DIR_PATH = '../model/'
ID = 'ID_code'
TARGET = 'target'
dtypes = setting.DTYPES


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
        print(f'already exist file path {filepath}')
    else:
        print('create profiling report')
        profile = pdp.ProfileReport(df)
        profile.to_file(filepath)


# oversampling train data
def smote_sampling(X, y):
    print(X.shape)
    print(y.shape)
    ds_values = y.value_counts()
    print(ds_values)
    # max_index = ds_values.idxmax()
    smote = SMOTE(sampling_strategy='auto', random_state=0, n_jobs=-1)
    X_resample, y_resample = smote.fit_resample(X, y)
    return X_resample, y_resample


# scaling data (numpy)
def scaling(data):
    data = StandardScaler().fit_transform(data)
    return data


def build_xgb_model():
    xgb_clf = xgb.XGBClassifier(n_estimators=100,
                                learning_rate=0.1,
                                max_depth=5,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                scale_pos_weight=1,
                                objective='binary:logistic',
                                # nthread=8,
                                silent=False
                                )
    return xgb_clf


# return two score
def model_score(y_true, y_pred):
    acc_score = str(accuracy_score(y_true, y_pred))
    class_report = str(classification_report(y_true, y_pred))
    print('accuracy score: ' + acc_score)
    print('classification report', class_report)

    return class_report, acc_score


if __name__ == '__main__':
    #
    # input data to dataframe
    #

    with multiprocessing.Pool() as pool:
        df_train, df_test = pool.map(load_file, ["train", "test"])

    # df_train = df_train.sample(2000)
    # df_test = df_test.sample(2000)
    print(df_train.info())
    print(df_test.info())

    output_profile(df_train, 'train.html')
    output_profile(df_test, 'test.html')

    features = [c for c in df_train.columns if c not in [ID, TARGET]]

    #
    # pre processing (oversampling and scaling)
    #
    x_train = df_train[features]
    y_train = df_train[TARGET]

    x_train, y_train = smote_sampling(x_train, y_train)

    x_train = scaling(x_train)
    x_test = scaling(df_test[features])

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=0)

    #
    # training model and save
    #

    clf = build_xgb_model()
    clf.fit(x_train, y_train)
    pickle.dump(clf, open(MODEL_DIR_PATH + 'model.sav', 'wb'))

    #
    # print predict and score
    #

    y_valid_pred = clf.predict(x_valid)
    acc_score, class_report = model_score(y_valid_pred, y_valid)

    #
    # print importance features
    #

    df_feat_imp = pd.Series(clf.feature_importances_, index=features)
    print(df_feat_imp.sort_values(ascending=False))

    #
    # test to predict and submit dataframe
    #

    y_test_pred = clf.predict(x_test)
    df_test_pred = pd.Series(y_test_pred, index=df_test[ID], name=TARGET)
    print(df_test_pred.head())
    print(df_test_pred.value_counts())

    df_test_pred.to_csv(SUBMIT_DIR_PATH + f'submit_{acc_score}.csv', header=True)
