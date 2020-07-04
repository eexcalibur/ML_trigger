from sklearn.metrics import confusion_matrix
import load_data
import my_metrics
import numpy as np
import plot_learning_curve
import sys 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

dataset = load_data.load_arm_hy("../../data/goamazon/trigger_goamazon_hy.nc")
#dilute_cape = np.loadtxt("/R0/wlin/Tao/cape/goamazom_cape.txt")
#dilute_lcl = np.loadtxt("/R0/wlin/Tao/cape/goamazom_lcl.txt")
dilute_cape = np.loadtxt("../../data/goamazon/goamazom_undilute_dcape.txt")
dilute_lcl = np.loadtxt("../..//data/goamazon/goamazom_undilute_lcl.txt")
#dilute_cape = np.loadtxt("/R0/wlin/Tao/cape/scam/cesm1_2_2/CAPE_case/run/goamazon_dilute_dcape.txt")
#dilute_lcl = np.loadtxt("/R0/wlin/Tao/cape/scam/cesm1_2_2/CAPE_case/run/goamazon_dilute_lcl.txt")

dataset['cape'] = dilute_cape
dataset['lcl'] = dilute_lcl

trig_x = dataset.iloc[:,0:86]
trig_y = dataset.iloc[:,86]

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
xgb = XGBClassifier(n_estimators=600,silent=True, nthread=8, max_depth=7,scale_pos_weight=3.5)

title = "Learning Curves (XGBoost)"
plot_learning_curve.plot_learning_curve(xgb,title,trig_x, trig_y,ylim=(0.7, 1.01), cv=cv, n_jobs=8)
plt.show()
