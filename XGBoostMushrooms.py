# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:00:24 2020

@author: User
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

mushrooms = pd.read_csv("mushrooms.csv")
poisonous = {"class": {"p":1, "e":0}}
mushrooms.replace(poisonous, inplace=True)
print(mushrooms.shape)
print(mushrooms.info())

X = mushrooms.iloc[:, 1:23]
y = mushrooms.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2, 
                                                    random_state=42)

X_train_oh = pd.get_dummies(X_train)
X_test_oh = pd.get_dummies(X_test)

# ======================== sKlearn style ============================
xgboost_sklearn_style = xgb.XGBClassifier(objective='binary:logistic',
                                           n_estimators=10, seed=42,
                                           n_jobs=-1)

xgboost_sklearn_style.fit(X_train_oh, y_train)
xg_preds = xgboost_sklearn_style.predict(X_test_oh)
# Compute the accuracy: accuracy
print(xgboost_sklearn_style.score(X_train_oh, y_train))
accuracy = float(np.sum(xg_preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# ======================== native style ============================
dtrain = xgb.DMatrix(data=X_train_oh, label=y_train)
dtest = xgb.DMatrix(data=X_test_oh, label=y_test)

print("Train dataset: {0} rows, {1} columns".format(dtrain.num_row(), 
      dtrain.num_col()))
print("Test dataset: {0} rows, {1} columns".format(dtest.num_row(), 
      dtest.num_col()))

# 'objective' :'binary:logistic' --> classification problem
# 'max_depth':2 --> shallow single trees with no more than two levels
# 'silent':1 --> no output

params={'objective' :'binary:logistic',
        'max_depth':2,
        'silent':1,
        'eta':1
        }
# iterate five rounds
num_rounds = 5

watchlist = [(dtest, 'test'), (dtrain, 'train')]
xgb_native = xgb.train(params, dtrain, num_rounds,watchlist)

pred_native = xgb_native.predict(dtest)
labels = dtest.get_label()
preds=pred_native > 0.5
correct =0

for i in range(len(preds)):
    if (labels[i] ==preds[i]):
        correct+=1

print("Predicted correctly {0}/{1}".format(correct, len(pred_native)))
print("Error:{0:.4f}".format(1-correct/len(preds)))
print("Accuracy:{0:.4f}".format(correct/len(pred_native))) # 0.9914

# ============================ FEATURE IMPORTANCES ==================================
# print(xgb_native.score(X_train_oh, y_train)) --> ERROR
xgb.plot_importance(xgb_native, importance_type='gain', xlabel='Gain')
# Using only f-score
xgb.plot_importance(xgb_native)
print(xgboost_sklearn_style.feature_importances_)
importances = pd.Series(data=xgboost_sklearn_style.feature_importances_,
                        index=X_train_oh.columns).sort_values(ascending=False)
# Drop zero values
importances = importances[importances!=0]
print(importances.index.values)

importances.sort_values().plot.barh(color='lightgreen')
plt.title('Features Importances')
plt.show()

# ==================== Train XGBOOST using only important features ===================
X_train_important = X_train_oh[importances.index.values]
print(X_train_important.info())
xgb_important = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, 
                                  seed=42, n_jobs=-1)
xgb_important.fit(X_train_important, y_train)
# predict on all
xgb_preds_important = xgb_important.predict(X_test_oh[importances.index.values])
# Compute the accuracy: accuracy
accuracy = float(np.sum(xgb_preds_important==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy)) 
# 0.981538 --> same as with all features

