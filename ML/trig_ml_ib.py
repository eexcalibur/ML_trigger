from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import load_data
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import f1_score

score_cv_df = pd.DataFrame()

def vert_hierarchy(var):
    ntime = var.shape[0]
    var_mean = np.zeros((ntime,3))
    var_mean[:,0] = np.mean(var[:,0:11], axis=1)
    var_mean[:,1] = np.mean(var[:,11:22], axis=1)
    var_mean[:,2] = np.mean(var[:,22:34], axis=1)
    #print(var_mean.shape)
    return var_mean

def hss_score(a,b,c,d):
    score = 2.0 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d)) 
    return score    

def seds_score(a,b,c,d):
    n = a + b + c + d 
    score = (np.log((a+b)/n) + np.log((a+c)/n))/np.log(a/n) - 1 
    return score

def ets_score(a,b,c,d):
    n = a + b + c + d 
    score = (a - (a + b) * (a + c)/n) / (a + b + c - (a + b) * (a + c)/n)
    return score

def my_scorer(estimator, x, y): 
    global score_cv_df
    y_pred = estimator.predict(x)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    a = tp
    b = fp
    c = fn
    d = tn  
    n = a + b + c + d 
    hss  = hss_score(a,b,c,d)
    seds = seds_score(a,b,c,d)
    ets  = ets_score(a,b,c,d)
    f1   = metrics.f1_score(y, y_pred, average='macro')
    score_df = pd.DataFrame({'hss':[hss], 'seds':[seds], 'ets':[ets], 'f1':[f1]})
    score_cv_df = pd.concat([score_cv_df, score_df])
    return f1

#dataset = load_data.load_sgp_data("trigger_sgp_hy.nc")
#dataset = load_data.load_twp_data()
#dataset = load_data.load_goamazon_data()
#dataset = load_data.load_arm_hy("trigger_goamazon_hy.nc")
dataset = load_data.load_era_obs("../../data/goamazon/goamazon_2014_to_2015_obs.nc")

print(dataset.shape)
#dataset.drop(['cape', 'lcl'], axis=1, inplace=True)

trig_x = dataset.iloc[:,0:85]
trig_y = dataset.iloc[:,85]

target_counts = dataset.label.value_counts()
print(target_counts)


#scaler = MinMaxScaler(feature_range=(0, 1))
#trig_x = scaler.fit_transform(trig_x)
scaler = StandardScaler()
scaler.fit(trig_x)
trig_x = scaler.transform(trig_x)

#train and test data
trig_x_train,trig_x_test,trig_y_train,trig_y_test= train_test_split(trig_x, trig_y, test_size=0.1, random_state=20)
smt = SMOTETomek(ratio='auto')
trig_x1, trig_y1 = smt.fit_sample(trig_x_train, trig_y_train)
print(sum(trig_y1))
print(trig_y1.shape)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)

#****************************************************************
#create and fit an xgboost classifier
#****************************************************************
print("xgboost")
xgb = XGBClassifier(n_estimators=600,max_depth=6, nthread=8)
xgb.fit(trig_x1, trig_y1)
trig_y_pred = xgb.predict(trig_x_test)
f1 = f1_score(trig_y_test, trig_y_pred, average='macro')
print(f1)
#f1_scores = cross_val_score(xgb, trig_x, trig_y, cv=cv, scoring=my_scorer)

#print(score_cv_df)
#print(score_cv_df.mean())


