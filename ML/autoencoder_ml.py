import autoencoder as ac
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

features,labels =  ac.build_autoencoder()

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)

print("xgboost")
xgb = XGBClassifier(n_estimators=2000,max_depth=6, nthread=4)
f1_scores = cross_val_score(xgb, features, labels, cv=cv, scoring="f1_macro")

print(f1_scores)
