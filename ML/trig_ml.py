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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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

if __name__ == '__main__':
    #dataset = load_data.load_sgp1()
    #dataset = load_data.load_arm_hy("trigger_goamazon_hy.nc")
    #dataset = load_data.load_twp_data()
    #dataset = load_data.load_goamazon_data()
    #dataset = load_data.load_era_obs("../../data/goamazon/goamazon_2014_to_2015_obs.nc")
    #dataset = load_data.load_era_obs("../../data/twp/twp_2014_to_2015_obs.nc")
    #dataset = load_data.load_sgp_data("trigger_sgp_hy.nc")
    #dataset = load_data.load_forcing("../../data/goamazon/GOAMAZON_iopfile_4scam.nc")
    #print(dataset.dtypes)
    #print(dataset.shape)
    #trig_x = dataset.iloc[:,2:148]
    #trig_y = dataset.iloc[:,1]
    
    dataset = load_data.load_sgp_data("trigger_sgp_hy.nc")
    print(dataset.dtypes)
    print(dataset.shape)
    trig_x = dataset.iloc[:,0:86]
    trig_y = dataset.iloc[:,86]

    #scaler = MinMaxScaler(feature_range=(0, 1))
    #trig_x = scaler.fit_transform(trig_x1)
    #scaler = StandardScaler()
    #scaler.fit(trig_x)
    #trig_x = scaler.transform(trig_x)
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
    
    #****************************************************************
    #create and fit an xgboost classifier
    #****************************************************************
    print("xgboost")
    sw = 0
    xgb = XGBClassifier(n_estimators=600,silent=True, nthread=8, max_depth=7, scale_pos_weight=3.5)
    #> cross validation
    if sw == 0:
        xgb = XGBClassifier(n_estimators=600,silent=True, nthread=8, max_depth=7, scale_pos_weight=3.5)
        f1_scores = cross_val_score(xgb, trig_x, trig_y, cv=cv, scoring=my_scorer)
        print(score_cv_df)
        print(score_cv_df.mean())

    #> composite analysis
    trig_x_train,trig_x_test,trig_y_train,trig_y_test= train_test_split(trig_x, trig_y, test_size=0.2)
    #>> machine learning
    if sw == 1:
        xgb.fit(trig_x_train,trig_y_train)
        trig_y_pred = xgb.predict(trig_x_test)
        tn, fp, fn, tp = confusion_matrix(trig_y_test, trig_y_pred).ravel()
        f1 = f1_score(trig_y_test, trig_y_pred,average='macro')
        print(tn,fp,fn,tp)
        print(f1)
  
    sys.exit()
    #>> dilute cape
    print(trig_x_test.shape)
    dilute_cape = trig_x_test.loc[:,"cape"].tolist()
    ntime = len(dilute_cape)
    dlabel = np.zeros(ntime)
    for i in range(ntime):
        if dilute_cape[i] > 70:
            dlabel[i] = 1
        else:
            dlabel[i] = 0

    tn, fp, fn, tp = confusion_matrix(trig_y_test, dlabel).ravel()
    f1 = f1_score(trig_y_test, dlabel,average='macro')
    print(tn,fp,fn,tp)
    print(f1)
    a = tp
    b = fp
    c = fn
    d = tn 
    hss  = hss_score(a,b,c,d)
    seds = seds_score(a,b,c,d)
    ets  = ets_score(a,b,c,d)

    print(hss,seds,ets)
