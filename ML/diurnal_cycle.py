import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

trig_df = load_data.load_sgp_data("trigger_sgp_hy.nc")

#machine learning predict
ML_dataset = trig_df.iloc[:,0:46]
scaler = StandardScaler()
scaler.fit(ML_dataset)
trig_x = scaler.transform(ML_dataset)
trig_df.iloc[:,0:46] = trig_x
trig_y = trig_df.iloc[:,46]

trig_x_train,trig_x_test,trig_y_train,trig_y_test= train_test_split(trig_df, trig_y, test_size=0.1, random_state=10)

xgb = XGBClassifier(n_estimators=600,max_depth=5, nthread=4)
xgb.fit(trig_x_train.iloc[:,0:46], trig_y_train)
pred = xgb.predict(trig_x_test.iloc[:,0:46])

trig_x_test['pred'] = pred


diurnal_prect = np.zeros(24)
diurnal_trig  = np.zeros(24)
for i in range(24):
    diurnal_prect[i] = trig_x_test[trig_x_test.hour == i].mean().PRECT
    diurnal_trig[i]  = trig_x_test[trig_x_test.hour == i].mean().pred

plt.plot(range(24), diurnal_prect, 'ro-', label="PRECT")
plt.bar(range(24), diurnal_trig, label="Trigger")
plt.legend()
plt.show()

