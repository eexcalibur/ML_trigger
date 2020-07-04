from sklearn.metrics import confusion_matrix
import numpy as np
import sys
from sklearn import metrics
from sklearn.metrics import f1_score

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
