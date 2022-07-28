import pandas as pd
import numpy as np
import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, r2_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import util
import os
import pickle


train_df = pd.read_csv('data/train.csv', delimiter="\t")
train_df = train_df.fillna(-1)

y_train = train_df.loc[:, 'label']
X_train = train_df.loc[:, train_df.columns != 'label']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.7, random_state=42)
model = XGBClassifier(max_depth=6, min_child_weight=1, n_estimators=100, scale_pos_weight=0.08, objective="binary:logistic")
model.fit(X_train, y_train)
joblib.dump(model, 'data/xgb.pkl')

test_df = pd.readcsv('data/test.csv', delimiter="\t")
test_df = test_df.fillna(-1)
y_test = test_df.loc[:, 'label']
X_test = test_df.loc[:, test_df.columns != 'label']

print('Simple model validation AUC: {:.4}'.format(
    roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
))

print('Simple model validation AUC: {:.4}'.format(
    roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
))
util.PlotKS(model.predict_proba(X_test)[:,1], y_test, 100, 0)
importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)


def score(df):
    print("load model begin")
    base_path = os.path.split(os.path.realpath(__file__))[0]
    fp = open(base_path + '/deploy/catboost.pkl', 'rb')
    model = pickle.load(fp)

    df = df.fillna(-1)

    category_columns = ['device_type', 'city', 'city_level', 'province', 'install_channel', 'risk_interesting_score',
                        'intersting_word', 'visit_frequency_m3']
    for col in category_columns:
        df[col] = df[col].astype('str')

    result = pd.DataFrame({});
    result['mid'] = df['md5']
    rm_cols = ['device_id', 'md5', 'qd_credit_date', 'qd_order_date', 'register_date']
    df = df.drop(rm_cols, axis=1)

    print("predict begin")
    result['prob'] = model.predict_proba(df)[:, 1]
    print("predict end")

    result.to_csv('member_score_B.csv', sep=',', header=False, index=False)


