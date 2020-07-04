import sys 
from sklearn.metrics import confusion_matrix
import load_data
import my_metrics
import numpy as np
import plot_learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
#from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score,precision_score,recall_score
import seaborn as sns
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import shap
import xgboost as xgb 

dilute_cape = np.loadtxt("../../data/goamazon/dilute_cape")
dilute_dcape = np.loadtxt("../../data/goamazon/dilute_dcape")
undilute_dcape = np.loadtxt("../../data/goamazon/undilute_dcape")
undilute_cape = np.loadtxt("../../data/goamazon/undilute_cape")
lcl = np.loadtxt("../../data/goamazon/dilute_lcl")

dataset = load_data.load_arm_hy("../../data/goamazon/trigger_goamazon_hy.nc")

dataset['cape'] = dilute_dcape
dataset['lcl'] = lcl
print(dataset.shape)
#print(dataset.dtypes)
dataset['PRECT'].plot()

pos = dataset[dataset.label==1]
neg = dataset[dataset.label==0]
trig_x = dataset.iloc[:,0:86]
trig_y = dataset.iloc[:,86]

trig_x_train,trig_x_test,trig_y_train,trig_y_test= train_test_split(trig_x, trig_y, test_size=0.2, random_state=20)


param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'binary:hinge'  # error evaluation for multiclass training
}  # the number of classes that exist in this datset
num_round = 600  # the number of training iterations
bst = xgb.train(param, xgb.DMatrix(trig_x, label=trig_y), num_round)

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(trig_x)

vars_name=['cape','lhflx','Tsair']

for var in vars_name:
    shap.dependence_plot(var, shap_values, trig_x,interaction_index=None)
