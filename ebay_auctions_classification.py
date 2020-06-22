#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:51:41 2020

@author: mrpaolo
"""

import pandas as pd
import numpy as np
from sklearn import pipeline,preprocessing,metrics,model_selection,ensemble
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_absolute_error


auctions = pd.read_csv("ebay_auctions.csv", error_bad_lines=False)
auctions.info()
data = auctions.copy()



auctions.head()
# ======================================================= FEATURE ENGINEERING =============================================================

# ----------------- CATEGORICAL DATA --------------------------

# -------------------------------------------------- One hot encoding Categories -------------------------------------
dummy_cats = pd.get_dummies(data['Category'])
data = pd.concat([data, dummy_cats], axis=1)
data.drop('Category', axis=1, inplace=True)
data.head()

# ------------------------------------------ Dealing with currencies --------------------------------------------
categories_encoder = preprocessing.LabelEncoder()
categories_encoder.fit(auctions['Category'])
#data['Category'] = categories_encoder.fit_transform(auctions['Category'])
print(categories_encoder.classes_) # ['EUR' 'GBP' 'US']
data['Currency'].head()

# ------------------------------------------ Converting days to numbers -----------------------------------
dayz_of_wik = {"End Day": {"Mon":1, "Tue":2, "Wed":3, "Thu":4, "Fri":5, "Sat":6, "Sun":7}}
data.replace(dayz_of_wik, inplace=True)
data['End Day'].head()

# ------------------------------------------ Dealing with currencies --------------------------------------------
currency_encoder = preprocessing.LabelEncoder()
data['Currency'] = currency_encoder.fit_transform(data['Currency'])
print(currency_encoder.classes_) # ['EUR' 'GBP' 'US']
data['Currency'].head()

# DEALING WITH NUMERICAL DATA
# ========================================== Numerical Columns =============================================================

# StandardScaler uses z-score with normalization

# --------------------- Close Price -------------------------------
close_price_scaler = preprocessing.StandardScaler() 
data['Close Price'] = close_price_scaler.fit_transform(data['Close Price'].values.reshape(-1,1))
print(close_price_scaler.mean_)# ~~0
data['Close Price'].head()

# --------------------- Other Columns -------------------------------
scaler = preprocessing.StandardScaler()
print(data[['Seller Rating','Duration','Open Price']].head())

data[['Seller Rating','Duration','Open Price']] = scaler.fit_transform(
        data[['Seller Rating','Duration','Open Price']])
print(scaler.mean_)


# ================================================ XGBOOST CLASSIFICATION ==========================================
data.info()
X = data.iloc[:, [0,1,2,3,4,5]] # Removing Toys/Hobbies and Collectibles from X values to test the hypotezis
y = data.iloc[:, 6]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
xgboost_competitive = xgb.XGBClassifier(objective='binary:logistic', n_estimators=900, max_depth=5, gamma=0.4, 
                                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, seed=27, n_jobs=-1)
xgboost_competitive.fit(X_train, y_train)
xg_predict = xgboost_competitive.predict(X_test)

# Seems like category doesn't influence the accuracy of prediction
# THUS: Results of auctions will be the same in all categories
print(xgboost_competitive.score(X_test, y_test))
accuracy = float(np.sum(xg_predict==y_test))/y_test.shape[0]
# One Hot Encoding Doesn't help with the accuracy of the model
print("accuracy: %f" % (accuracy))# accuracy: 0.76666666 --> tuchliak

# --------------------------- FEATURE IMPORTANCES -----------------------------------
for imp_type in ['weight', 'gain', 'cover']:
    xgb.plot_importance(xgboost_competitive, importance_type=imp_type, xlabel=str(imp_type).capitalize())

importances = pd.Series(data=xgboost_competitive.feature_importances_, index=X_train.columns).sort_values(ascending=False)
# Drop zero values
importances = importances[importances!=0]
print(importances.index.values)

importances.sort_values().plot.barh(color='lightgreen')
plt.title('Features Importances')
plt.show()

# ------------------------------------ XGBOOST HYPERPARAMETER TUNING ------------------------------

param_test1 = {
 'max_depth': range(4, 9, 1),
 'learning_rate':np.linspace(0.1, 10, num=10),
 'n_estimators': range(100, 1200, 200),
 'gamma':[i/10.0 for i in range(0,5)]
}

param_test2 = {
 'subsample':np.linspace(0.1, 1.0, num=10),
 'colsample_bytree':np.linspace(0.1, 1.0, num=10)
}

gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',iid=False, cv=5, n_jobs=-1)
gsearch1.fit(X_train,y_train)
print(gsearch1.cv_results_ , gsearch1.best_params_, gsearch1.best_score_)
print("\nBest Parameters:",gsearch1.best_params_, gsearch1.best_score_)
# Best Parameter: {'max_depth': 5, 'min_child_weight': 1} 0.7872227772227771
# Best Parameters: {'learning_rate': 0.1} 0.7872227772227771
# Best Parameters: {'gamma': 0.4, 'learning_rate': 0.1, 'n_estimators': 900} 0.7992407592407592

# ====================================================== XGBOOST REGRESSION ======================================================
print(data.info())
X = data.iloc[:, [0,1,2,3,5,6,7,8,9]]
X.info()
y = data.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
xgboost_regression = xgb.XGBRegressor(colsample_bytree=0.4, gamma=0, learning_rate=0.1,
                 max_depth=5, min_child_weight=1, n_estimators=900, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=27)

xgboost_regression.fit(X_train, y_train)
xg_predict = xgboost_regression.predict(X_test)

mean_absolute_error(y_test, xg_predict ) #  0.4967 --> meaning we are 0.5% off when predicting the close price


# --------------------------- FEATURE IMPORTANCES -----------------------------------
for imp_type in ['weight', 'gain', 'cover']:
    xgb.plot_importance(xgboost_regression, importance_type=imp_type, xlabel=str(imp_type).capitalize())

# Seller Rating and Open Price are the most influencial features
    

# ++++++++++++++++++++++++++++++++++++++++++ XGBOOST REGRESSION ON THE WHOLE DATASET +++++++++++++++++++++++++++++++++++++++
X_full = data.iloc[:, [0,1,2,3,5,6,7,8,9]]
y_full = data.iloc[:, 4]

xgboost_regression_full = xgb.XGBRegressor(colsample_bytree=0.4, gamma=0, learning_rate=0.1,
                 max_depth=5, min_child_weight=1, n_estimators=900, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=27)

xgboost_regression_full.fit(X_full, y_full)

X_reverse = scaler.inverse_transform(X_full)

xgb_close_price = xgboost_regression_full.predict(X_full)

mean_absolute_error(y_full, xgb_close_price ) #  0.1116

close_price_forecast = pd.DataFrame({"Close Price Forecast": xgb_close_price})
data = pd.concat([data, close_price_forecast], axis=1)
print(auctions.info())

# ++++++++++++++++++++++++++++++++++++++++++++++++ WITHOUT STANDARD SCALING ++++++++++++++++++++++++++++++++++++++++++++++++++
dayz_of_wik = {"End Day": {"Mon":1, "Tue":2, "Wed":3, "Thu":4, "Fri":5, "Sat":6, "Sun":7}}
data.replace(dayz_of_wik, inplace=True)

label_encoder = preprocessing.LabelEncoder()
data[['Currency', 'Category']] = data[['Currency', 'Category']].apply(preprocessing.LabelEncoder().fit_transform)
data.head()

print(auctions.info())
X_unsc = auctions.iloc[:, [0,1,2,3,4,6,7]]
X_unsc.info()
y_unsc = auctions.iloc[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X_unsc,y_unsc, test_size=0.2, random_state=42)
xgb_reg_unscaled = xgb.XGBRegressor(colsample_bytree=0.4, gamma=0, learning_rate=0.1,
                 max_depth=5, min_child_weight=1, n_estimators=900, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=27)

xgb_reg_unscaled.fit(X_train, y_train)
xgb_pred_unscaled = xgb_reg_unscaled.predict(X_test)

mean_absolute_error(y_test, xgb_pred_unscaled )# 55.0596 --> Terrible
close_price_forecast = pd.DataFrame({"Close Price Forecast": xgb_pred_unscaled})
auctions = pd.concat([auctions, close_price_forecast], axis=1)

# =================================================== WITH PIPELINE ==========================================================
import pandas as pd
import numpy as np
from sklearn import pipeline, preprocessing, metrics, model_selection, ensemble
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import mean_absolute_error

auctions = pd.read_csv("ebay_auctions.csv")
dayz_of_wik = {"End Day": {"Mon":1, "Tue":2, "Wed":3, "Thu":4, "Fri":5, "Sat":6, "Sun":7}}
auctions.replace(dayz_of_wik, inplace=True)

close_price_scaler = preprocessing.StandardScaler() 
auctions['Close Price'] = close_price_scaler.fit_transform(auctions['Close Price'].values.reshape(-1,1))
print(close_price_scaler.mean_)# ~~0
auctions['Close Price'].head()

print(auctions.info())

mapper = DataFrameMapper([
        (['Seller Rating','Duration','Open Price'], preprocessing.StandardScaler()),
        (['Currency'],preprocessing.LabelEncoder()),
        (['Category'],preprocessing.LabelEncoder())
        ])

pipeline_obj = pipeline.Pipeline([
    ('mapper',mapper),
    ("model", xgb.XGBRegressor(colsample_bytree=0.4, gamma=0, learning_rate=0.1, max_depth=5, min_child_weight=1, 
                               n_estimators=900, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=27))
])
    

#X=['Seller Rating','Duration','Competitive','Open Price', 'Currency', 'Category','End Day']
#y=['Close Price']

X = auctions.iloc[:, [0,1,2,3,4,6,7]]
X.info()
y = auctions.iloc[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

pipeline_obj.fit(X_train,y_train.ravel())
y_pred = pipeline_obj.predict(X_test)
mean_absolute_error(y_test, y_pred)

# ============================================= Let's try to predict the price =============================================
auctions.info()

dayz_of_wik = {"End Day": {"Mon":1, "Tue":2, "Wed":3, "Thu":4, "Fri":5, "Sat":6, "Sun":7}}
data.replace(dayz_of_wik, inplace=True)

# ATTENTION: Names of the columns should be transferred as a list to use 1-D array
auction_lot_mapper = [
    ('Category', categories_encoder),
    ('Currency', currency_encoder),
    (['Seller Rating'], scaler),
    (['Duration'], scaler),
    (['Open Price'], scaler)  
]

auction_lot_mapper = [
    ('Category', preprocessing.LabelEncoder()),
    ('Currency', preprocessing.LabelEncoder()),
    (['Seller Rating'], preprocessing.StandardScaler()),
    (['Duration'], preprocessing.StandardScaler()),
    (['Open Price'], preprocessing.StandardScaler())  
]

mapper = DataFrameMapper(auction_lot_mapper, df_out=True)

auctions["End Day"].value_counts()
my_lot = {"Category": 'Collectibles',
          "Currency": 'US',
          "Seller Rating": [5000],
          "Duration": [7],
          "End Day": 'Mon',
          "Open Price": [7.00], 
          "Competitive": [1]
          }


my_lot_df = pd.DataFrame.from_dict(my_lot)
my_lot_df.info()

currency_encoder.fit_transform(my_lot_df['Currency'])
mapper_fit = mapper.fit(my_lot_df)


train_processed = mapper.transform(my_lot_df)
print(train_processed.head())

# ================================================= PICKLE THE MODEL ==================================================

from sklearn.externals import joblib

joblib.dump(pipeline_obj,'RFModelforMPG.pkl')

modelReload=joblib.load('RFModelforMPG.pkl')




