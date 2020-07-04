import sys
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import load_data

def single_factor_diag(pos, neg, name, figname):
   fac_pos = pos[name]
   fac_neg = neg[name]
   box_data = pd.concat([fac_pos, fac_neg], axis=1)
   box_data.columns = ['Convection', 'No Convection']
   #box_data.boxplot(grid=False)
   box_data.plot.kde()
   #plt.title("Tsair")
   #plt.xlim(280,320)
   #plt.savefig(figname)
   plt.show()
   sys.exit()

def profile_anom_confusion(trig_df):
    nlev = 36
    slev = 6
    trig_df_mean = trig_df.mean()
    T_mean = trig_df_mean[slev:slev+nlev]
    Q_mean = trig_df_mean[slev+nlev:slev+nlev*2]
    divs_mean = trig_df_mean[slev+nlev*2:slev+nlev*3]
    divq_mean = trig_df_mean[slev+nlev*3:slev+nlev*4]    

    trig_pos_df = trig_df[trig_df.label == 1]
    trig_neg_df = trig_df[trig_df.label == 0]

    T_pos_anom = trig_pos_df.iloc[:,slev:slev+nlev] - T_mean
    T_neg_anom = trig_neg_df.iloc[:,slev:slev+nlev] - T_mean
    Q_pos_anom = trig_pos_df.iloc[:,slev+nlev:slev+nlev*2] - Q_mean
    Q_neg_anom = trig_neg_df.iloc[:,slev+nlev:slev+nlev*2] - Q_mean
    divs_pos_anom = trig_pos_df.iloc[:,slev+nlev*2:slev+nlev*3] - divs_mean
    divs_neg_anom = trig_neg_df.iloc[:,slev+nlev*2:slev+nlev*3] - divs_mean
    divq_pos_anom = trig_pos_df.iloc[:,slev+nlev*3:slev+nlev*4] - divq_mean
    divq_neg_anom = trig_neg_df.iloc[:,slev+nlev*3:slev+nlev*4] - divq_mean
    
    T_pos_anom_mean = np.mean(T_pos_anom, axis=0)
    T_neg_anom_mean = np.mean(T_neg_anom, axis=0)
    Q_pos_anom_mean = np.mean(Q_pos_anom, axis=0) * 1000.0
    Q_neg_anom_mean = np.mean(Q_neg_anom, axis=0) * 1000.0
    divs_pos_anom_mean = np.mean(divs_pos_anom, axis=0) 
    divs_neg_anom_mean = np.mean(divs_neg_anom, axis=0)
    divq_pos_anom_mean = np.mean(divq_pos_anom, axis=0) 
    divq_neg_anom_mean = np.mean(divq_neg_anom, axis=0)
    
    T_pos_anom_std = np.std(T_pos_anom, axis=0)
    T_neg_anom_std = np.std(T_neg_anom, axis=0)
    Q_pos_anom_std = np.std(Q_pos_anom, axis=0)
    Q_neg_anom_std = np.std(Q_neg_anom, axis=0)


    lev = np.loadtxt("lev1.txt")
    print(T_pos_anom_mean)

    plt.subplot(221)
    plt.plot(T_pos_anom_mean, lev, label="Convection")
    plt.plot(T_neg_anom_mean, lev, label="No Convection")
    plt.legend()
    plt.title("Temperature")
    plt.gca().invert_yaxis()

    plt.subplot(222)
    plt.plot(Q_pos_anom_mean, lev, label="Convection")
    plt.plot(Q_neg_anom_mean, lev, label="No Convetion")
    plt.legend()
    plt.title("Humidity")
    plt.gca().invert_yaxis()

    plt.subplot(223)
    plt.plot(divs_pos_anom_mean, lev, label="Convection")
    plt.plot(divs_neg_anom_mean, lev, label="No Convection")
    plt.legend()
    plt.title("DIVS")
    plt.gca().invert_yaxis()

    plt.subplot(224)
    plt.plot(divq_pos_anom_mean, lev, label="Convection")
    plt.plot(divq_neg_anom_mean, lev, label="No Convection")
    plt.legend()
    plt.title("DIVQ")
    plt.gca().invert_yaxis()

    plt.savefig("profile_cmp_confusion")
    plt.show()
    
def profile_anom_clustering(trig_df):
    start = 4
    end   = 24
    df_mean = trig_df.mean().iloc[start:end]

    trig_pos_df = trig_df[trig_df.label == 1]
    trig_neg_df = trig_df[trig_df.label == 0]

    kmeans_pos = KMeans(n_clusters=11, random_state=0, n_jobs=8).fit(trig_pos_df)   
    kmeans_neg = KMeans(n_clusters=13, random_state=0, n_jobs=8).fit(trig_neg_df) 
    label_pos  = kmeans_pos.labels_
    label_neg  = kmeans_neg.labels_
  
    trig_pos_df['kmeans'] = label_pos
    trig_neg_df['kmeans'] = label_neg

    trig_pos_anom_1 = np.mean(trig_pos_df[trig_pos_df.kmeans == 0].iloc[:,start:end] - df_mean)
    trig_pos_anom_2 = np.mean(trig_pos_df[trig_pos_df.kmeans == 4].iloc[:,start:end] - df_mean)
    trig_pos_anom_3 = np.mean(trig_pos_df[trig_pos_df.kmeans == 5].iloc[:,start:end] - df_mean)
    trig_pos_anom_4 = np.mean(trig_pos_df[trig_pos_df.kmeans == 6].iloc[:,start:end] - df_mean)

    trig_neg_anom_1 = np.mean(trig_neg_df[trig_neg_df.kmeans == 8].iloc[:,start:end] - df_mean)
    trig_neg_anom_2 = np.mean(trig_neg_df[trig_neg_df.kmeans == 7].iloc[:,start:end] - df_mean)
    trig_neg_anom_3 = np.mean(trig_neg_df[trig_neg_df.kmeans == 3].iloc[:,start:end] - df_mean)
    trig_neg_anom_4 = np.mean(trig_neg_df[trig_neg_df.kmeans == 0].iloc[:,start:end] - df_mean)

    lev = np.loadtxt("lev.txt")

    plt.plot(trig_pos_anom_1, lev, label="Pos_1")
    plt.plot(trig_pos_anom_2, lev, label="Pos_2")
    plt.plot(trig_pos_anom_3, lev, label="Pos_3")
    plt.plot(trig_pos_anom_4, lev, label="Pos_4")
    
    plt.plot(trig_neg_anom_1, lev, label="Neg_1")
    plt.plot(trig_neg_anom_2, lev, label="Neg_2")
    plt.plot(trig_neg_anom_3, lev, label="Neg_3")
    plt.plot(trig_neg_anom_4, lev, label="Neg_4")
    plt.legend()
    plt.title("Temperature")
    plt.gca().invert_yaxis()
    plt.show()
    plt.savefig("profile_clustering")
#****************************************************************
#read postive and negative data
#****************************************************************

#trig_dataset_df = load_data.load_arm_hy("trigger_goamazon_hy.nc")
trig_dataset_df = load_data.load_goamazon_data()

##split into positive and negative
trig_pos_df = trig_dataset_df[trig_dataset_df.label == 1]
trig_neg_df = trig_dataset_df[trig_dataset_df.label == 0]
size_pos = trig_pos_df.shape
size_neg = trig_neg_df.shape
print(trig_pos_df.shape)
print(trig_neg_df.shape)

#for RHair
single_factor_diag(trig_pos_df, trig_neg_df, 'RHair', 'RHair_hist')

#for CAPE
#single_factor_diag(trig_pos_df, trig_neg_df, 'cape', 'CAPE_hist')

#for Tsair
#single_factor_diag(trig_pos_df, trig_neg_df, 'Tsair', 'Tsair_hist')

#for lhflx
#single_factor_diag(trig_pos_df, trig_neg_df, 'lhflx', 'lhflx_box')

#for shflx
#single_factor_diag(trig_pos_df, trig_neg_df, 'shflx', 'shflx_box')

#for T and Q
#profile_anom_confusion(trig_dataset_df)


#for joint RHair and Tsair
#sns.jointplot(x="RHair", y="PRECT", data=trig_dataset_df, kind="kde")
#plt.title("all")
#sns.jointplot(x="RHair", y="PRECT", data=trig_pos_df, kind="kde")
#plt.title("pos")
#sns.jointplot(x="RHair", y="PRECT", data=trig_neg_df, kind="kde")
#plt.title("neg")
#plt.savefig("joint_RHair_Tsair_pos")




#for clustering for pos and neg
#print("pos")
#kmeans = KMeans(n_clusters=11, random_state=0, n_jobs=8).fit(trig_pos_df)
#codebook = kmeans.cluster_centers_
#label    = kmeans.labels_
#trig_pos_df['kmeans'] = label
#
#ss = metrics.silhouette_score(trig_pos_df, label, metric='euclidean')
#print(ss)
#
#freq = np.zeros(11)
#for i in label:
#    freq[i] += 1
#
#print(freq)
##for i in [4,1,5]:
##    print(codebook[i, 3])
##
#print("neg")
#kmeans = KMeans(n_clusters=13, random_state=0, n_jobs=8).fit(trig_neg_df)
#codebook = kmeans.cluster_centers_
#label    = kmeans.labels_
#ss = metrics.silhouette_score(trig_neg_df, label, metric='euclidean')
#print(ss)
#
#freq = np.zeros(13)
#for i in label:
#    freq[i] += 1
#
#print(freq)
#profile_anom_confusion(trig_dataset_df)
#profile_anom_clustering(trig_dataset_df)
plt.show()
