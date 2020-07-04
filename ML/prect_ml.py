import load_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

dataset = load_data.load_data("trigger_1999_2008_summer_hy.nc")

prect_x = dataset.iloc[:,0:46]
prect_y = dataset.iloc[:,48]

scaler = StandardScaler()
scaler.fit(prect_x)
prect_x = scaler.transform(prect_x)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

#****************************************************************
#GBR
#****************************************************************
#print("GBR")
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#gbr = ensemble.GradientBoostingRegressor(**params)
#mse = cross_val_score(gbr, prect_x, prect_y, cv=cv, scoring='r2', n_jobs=5)
#
#print(mse)

