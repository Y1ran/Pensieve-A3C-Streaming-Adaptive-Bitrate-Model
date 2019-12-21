# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 14:30:55 2019

@author: Administrator
"""

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import pickle
from sklearn import datasets, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

data = pd.read_csv('.\TP_list.csv')
data.head()
#data = data.head(5000)
#X = data[['Current_buf', 'TP','Bit_rate','target_buf']]
#X = X.head(10000)
#X_bit0 = data#[data['Bit_rate']==0]
X_bit1 = data#[data['Bit_rate']==1]

#y_bit0 = X_bit0[['next_buf']]
y_bit1 = X_bit1[['next_size']]

#X_bit0 = X_bit0[['Current_buf', 'TP', 'Bit_rate','target_buf']]
X_bit1 = X_bit1[['data_size', 'bit_rate','TP']]

#y = y.head(10000)

X_train, X_test, y_train, y_test = train_test_split(X_bit1, y_bit1, test_size=0.05, random_state=1)
#
#
#linreg = LinearRegression()
#linreg.fit(X_train, y_train)
#
#y_pred = linreg.predict(X_test)
#
#print("MSE:",metrics.mean_squared_error(y_test, y_pred))
## 用scikit-learn计算RMSE
#print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print("Test set score:{:.2f}".format(linreg.score(X_test,y_test)))

'''CV-linreg'''
#predicted = cross_val_predict(linreg, X, y, cv=10)
#
## 用scikit-learn计算MSE
#print("MSE:",metrics.mean_squared_error(y, predicted))
## 用scikit-learn计算RMSE
#print("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))


'''Kernel_ridge'''
#regr=KernelRidge(kernel = 'poly')
#regr = GridSearchCV(regr, param_grid={"alpha":[0.1,1,5]}, cv=4)
##regr.fit(X_train,y_train)
##print(regr.best_estimator_)
#
#regr.fit(X_train,y_train) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
#print("Test set score:{:.2f}".format(regr.score(X_test,y_test)))
##print("Best parameters:{}".format(regr.best_params_))
##print("Best score on train set:{:.2f}".format(regr.best_score_))
#
#y_pred = regr.predict(X_test)
#print("MSE:",metrics.mean_squared_error(y_test, y_pred))

'''SVR'''
#svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                             "gamma": np.logspace(-2, 2, 5)})
#svr = SVR()
#svr.fit(X_train,y_train)
#print("Test set score:{:.2f}".format(svr.score(X_test,y_test)))
#print("Best score on train set:{:.2f}".format(svr.best_score_))
#y_pred = svr.predict(X_test)

'''lgb'''


#
gbm = ExtraTreeRegressor()

gbm = GridSearchCV(gbm, param_grid={"min_samples_leaf":[1,4,8,16,32],\
                                     'min_samples_split':[4,10,20,100],\
                                  'max_depth':[2,8,16,32]}, cv=6)
                                            
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
# eval
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
print("Test set score:{:.2f}".format(gbm.score(X_test,y_test)))
#print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_pred))


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()],[y_pred.min(), y_pred.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

with open('model.pickle', 'wb') as fw:
    pickle.dump(gbm, fw)
#with open('model.pickle', 'rb') as fr:
#    new_svm = pickle.load(fr)
#    print(new_svm.predict(X[0:1]))