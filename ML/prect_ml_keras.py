import load_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import keras

dataset = load_data.load_data("trigger_1999_2008_summer_hy.nc")

prect_x = dataset.iloc[:,0:46]
prect_y = dataset.iloc[:,48]

scaler = StandardScaler()
scaler.fit(prect_x)
prect_x = scaler.transform(prect_x)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

