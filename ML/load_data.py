import pandas as pd
from netCDF4 import Dataset,num2date
import numpy as np
import matplotlib.pyplot as plt
import sys
import mod_cape_v01_scam


def load_forcing(filename):
#****************************************************************
#read variables
#****************************************************************
    fid = Dataset(filename)
    ntime = fid.dimensions['time'].size
    #nlev  = fid.dimensions['lev'].size
    nlev  = 35
    lhflx = fid.variables['lhflx'][:].reshape(ntime,1)
    shflx = fid.variables['shflx'][:].reshape(ntime,1)
    PS    = fid.variables['Ps'][:].reshape(ntime,1)
    prect = fid.variables['prec_srf'][:].reshape(ntime,1)   #mm/h
    Tsair = fid.variables['Tg'][:].reshape(ntime,1) - 273.15 #K
    q_srf = fid.variables['qs'][:].reshape(ntime,1)
    T     = fid.variables['T'][:,5:40].reshape(ntime,nlev)
    q     = fid.variables['q'][:,5:40].reshape(ntime,nlev)
    divs  = fid.variables['divT'][:,5:40].reshape(ntime,nlev)
    divq  = fid.variables['divq'][:,5:40].reshape(ntime,nlev)
    lev   = fid.variables['lev'][5:40]

    np.savetxt("lev_goamazon.txt", lev/100.0)

    es = 6.112*np.exp(17.67*Tsair/(243.5+Tsair))
    qs = 0.622*es/(PS - es)
    RHsair = q_srf/qs


 #compute CAPE
    z  = 0.3048*(1 - (lev[:]/100/1013.25)**0.190284)*145366.45
    zs = 0.3048*(1 - (PS/100/1013.25)**0.190284)*145366.45

    ph    = np.zeros(ntime)
    cape  = np.zeros(ntime)
    cin   = np.zeros(ntime)
    lcl   = np.zeros(ntime)
    lfc   = np.zeros(ntime)
    lnb   = np.zeros(ntime)
    #for itime in range(ntime1):
    #    ph[itime], cape[itime], cin[itime], lcl[itime], lfc[itime], lnb[itime] = mod_cape_v01_scam.fcape_1d(lev[:]*100.0,t[itime,:], q[itime,:], z[:], zs[itime], 1)

    cape = cape.reshape(ntime,1)
    lcl  = lcl.reshape(ntime,1)

#set the label 
    prec_threshold = 0.5 
    label = np.zeros(ntime)
    for i in range(ntime):
        if(prect[i] >= prec_threshold):
            label[i] = 1 
        else:
            label[i] = 0 
    label = label.reshape(ntime,1)

    ncol = 8+nlev*4
    colums1 = ["" for x in range(ncol)]
  
    idx = 0
    for var in ['prect','label','lhflx','shflx','Tsair','RHsair','cape','lcl']:
        colums1[idx] = var
        idx += 1
  
    idx = 8
    for var in ['T','Q','divs','divq']:
        for i in lev:
            colums1[idx] = var+"_"+str(int(i/100.0))
            idx = idx + 1
 
    trig_dataset = np.concatenate((prect,label,lhflx,shflx,Tsair,RHsair,cape,lcl,T,q,divs,divq),axis=1)
    df = pd.DataFrame(trig_dataset, columns=colums1,dtype=np.float)
    return df


def load_era_trmm():
#****************************************************************
#read ERA-interim data
#****************************************************************
    #fid_era = Dataset("../../data/sgp/sgp_ERAI_2004_to_2013.nc")
    fid_era = Dataset("../../data/goamazon/goamazon_ERAI_2002_to_2015.nc")
    ntime1 = fid_era.dimensions['time'].size
    nlev1  = fid_era.dimensions['level'].size
    t2m = fid_era.variables['t2m'][:].reshape(ntime1,1)
    d2m = fid_era.variables['d2m'][:].reshape(ntime1,1)
    sp  = fid_era.variables['sp'][:].reshape(ntime1,1)
    t   = fid_era.variables['t'][:,:].reshape(ntime1,nlev1)
    q   = fid_era.variables['q'][:,:].reshape(ntime1,nlev1)
    r   = fid_era.variables['r'][:,:].reshape(ntime1,nlev1)
    w   = fid_era.variables['w'][:,:].reshape(ntime1,nlev1)
    lev = fid_era.variables['level'][:].reshape(nlev1)
    time= fid_era.variables['time'][:]
    time_unit = fid_era.variables['time'].units
    time_cal = fid_era.variables['time'].calendar

#compute CAPE
    z  = 0.3048*(1 - (lev[:]/1013.25)**0.190284)*145366.45
    zs = 0.3048*(1 - (sp/100/1013.25)**0.190284)*145366.45

    ph    = np.zeros(ntime1)
    cape  = np.zeros(ntime1)
    cin   = np.zeros(ntime1)
    lcl   = np.zeros(ntime1)
    lfc   = np.zeros(ntime1)
    lnb   = np.zeros(ntime1)
    for itime in range(ntime1):
        ph[itime], cape[itime], cin[itime], lcl[itime], lfc[itime], lnb[itime] = mod_cape_v01_scam.fcape_1d(lev[:]*100.0,t[itime,:], q[itime,:], z[:], zs[itime], 1)

    cape = cape.reshape(ntime1,1)
    lcl  = lcl.reshape(ntime1,1)

#set the label based on cape
    cape_threshold = 70
    clabel = np.zeros(ntime1)
    for i in range(ntime1):
        if(cape[i,0] >= cape_threshold):
            clabel[i] = 1
        else:
            clabel[i] = 0
    clabel = clabel.reshape(ntime1,1)

#format the date
    dates = num2date(time, time_unit, time_cal)
    year  = np.array([dt.strftime("%Y") for dt in dates]).reshape(ntime1,1)
    mon   = np.array([dt.strftime("%m") for dt in dates]).reshape(ntime1,1)
    day   = np.array([dt.strftime("%d") for dt in dates]).reshape(ntime1,1)
    hour  = np.array([dt.strftime("%H") for dt in dates]).reshape(ntime1,1)

    ncol = 10+nlev1*4
    colums1 = ["" for x in range(ncol)]

    idx = 0
    for var in ['year','mon','day','hour','t2m','d2m','sp','cape','lcl','clabel']:
        colums1[idx] = var
        idx += 1
    
    idx = 10
    for var in ['t','q','r','w']:
        for i in lev:
            colums1[idx] = var+"_"+str(i)
            idx = idx + 1

    trig_dataset1 = np.concatenate((year,mon,day,hour,t2m,d2m,sp,cape,lcl,clabel,t,q,r,w),axis=1)
    df1_temp = pd.DataFrame(trig_dataset1, columns=colums1,dtype=np.float)
    df1_temp.drop(['year','mon','day','hour'],axis=1, inplace=True)

#****************************************************************
#read trmm data    
#****************************************************************=
    #fid_trmm = Dataset("../../data/sgp/trmm.2004-2013.nc")
    fid_trmm = Dataset("../../data/goamazon/trmm.2002-2015.nc")
    ntime2 = fid_trmm.dimensions['time'].size
    #prect  = fid_trmm.variables['pcp'][:,3,3].reshape(ntime2,1)
    prect  = fid_trmm.variables['pcp'][:,28,1].reshape(ntime2,1)
    time2  = fid_trmm.variables['time'][:]
    time_unit2 = fid_trmm.variables['time'].units
    #time_cal2 = fid_trmm.variables['time'].calendar

    dates2 = num2date(time2, time_unit2)
    year2  = np.array([dt.strftime("%Y") for dt in dates2]).reshape(ntime2,1)
    mon2   = np.array([dt.strftime("%m") for dt in dates2]).reshape(ntime2,1)
    day2   = np.array([dt.strftime("%d") for dt in dates2]).reshape(ntime2,1)
    hour2  = np.array([dt.strftime("%H") for dt in dates2]).reshape(ntime2,1)

#set the label 
    prec_threshold = 0.5
    label = np.zeros(ntime2)
    for i in range(ntime2):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime2,1)

#set the column name
    colums2 = ["" for x in range(6)]
    idx = 0
    for var in ['label','prect','year','mon','day','hour']:
        colums2[idx] = var
        idx = idx + 1
    
    trig_dataset2 = np.concatenate((label,prect,year2,mon2,day2,hour2),axis=1)
    df2_temp = pd.DataFrame(trig_dataset2, columns=colums2, dtype=np.float)
    df2_temp = df2_temp[df2_temp.hour % 6 == 0].reset_index(drop=True)

    df = pd.concat([df1_temp, df2_temp], axis=1)
    df = df[df.prect >= 0]
    df = df.reset_index(drop=True)
    return df

def load_sgp():
#****************************************************************
#read ERA-interim data
#****************************************************************
    fid_era = Dataset("../../data/sgp/sgp_ERAI_2004_to_2005.nc")
    ntime1 = fid_era.dimensions['time'].size
    nlev1  = fid_era.dimensions['level'].size - 15
    t2m = fid_era.variables['t2m'][:].reshape(ntime1,1)
    d2m = fid_era.variables['d2m'][:].reshape(ntime1,1)
    sp  = fid_era.variables['sp'][:].reshape(ntime1,1)
    t   = fid_era.variables['t'][:,15:].reshape(ntime1,nlev1)
    q   = fid_era.variables['q'][:,15:].reshape(ntime1,nlev1)
    r   = fid_era.variables['r'][:,15:].reshape(ntime1,nlev1)
    w   = fid_era.variables['w'][:,15:].reshape(ntime1,nlev1)
    lev = fid_era.variables['level'][15:].reshape(nlev1)
    time= fid_era.variables['time'][:]
    time_unit = fid_era.variables['time'].units
    time_cal = fid_era.variables['time'].calendar

    dates = num2date(time, time_unit, time_cal)
    year  = np.array([dt.strftime("%Y") for dt in dates]).reshape(ntime1,1)
    mon   = np.array([dt.strftime("%m") for dt in dates]).reshape(ntime1,1)
    day   = np.array([dt.strftime("%d") for dt in dates]).reshape(ntime1,1)
    hour  = np.array([dt.strftime("%H") for dt in dates]).reshape(ntime1,1)

    ncol = 7+nlev1*4
    colums1 = ["" for x in range(ncol)]
    

    print(lev)

    idx = 0
    for var in ['year','mon','day','hour','t2m','d2m','sp']:
        colums1[idx] = var
        idx += 1
    
    idx = 7
    for var in ['t','q','r','w']:
        for i in lev:
            colums1[idx] = var+"_"+str(i)
            idx = idx + 1

    trig_dataset1 = np.concatenate((year,mon,day,hour,t2m,d2m,sp,t,q,r,w),axis=1)
    trig_dataset_df1 = pd.DataFrame(trig_dataset1, columns=colums1)
    df1_temp = trig_dataset_df1[(trig_dataset_df1.mon.astype(int)>=6) & (trig_dataset_df1.mon.astype(int)<= 8)].reset_index(drop=True)

#****************************************************************
#read forcing data    
#****************************************************************=
    fid_forc = Dataset("../../data/sgp/ContForcing_4scam/zhang_cps_trigger.nc")
    ntime2 = fid_forc.dimensions['time'].size
    lhflx  = fid_forc.variables['lhflx'][:].reshape(ntime2,1)
    shflx  = fid_forc.variables['shflx'][:].reshape(ntime2,1)
    prect  = fid_forc.variables['Prec'][:].reshape(ntime2,1) * 3600
    year   = fid_forc.variables['year'][:].reshape(ntime2,1)
    mon    = fid_forc.variables['month'][:].reshape(ntime2,1)
    day    = fid_forc.variables['day'][:].reshape(ntime2,1)
    hour   = fid_forc.variables['hour'][:].reshape(ntime2,1)

#set the label 
    prec_threshold = 0.5
    label = np.zeros(ntime2)
    for i in range(ntime2):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime2,1)


    colums2 = ["" for x in range(8)]
    idx = 0
    for var in ['year','mon','day','hour','lhflx','shflx','prect','label']:
        colums2[idx] = var
        idx = idx + 1
    
    trig_dataset2 = np.concatenate((year,mon,day,hour,lhflx,shflx,prect,label),axis=1)
    trig_dataset_df2 = pd.DataFrame(trig_dataset2, columns=colums2)
    df2_temp = trig_dataset_df2[(trig_dataset_df2.year == 2004) | (trig_dataset_df2.year == 2005)]
    df2_temp = df2_temp[df2_temp.hour % 6 == 0].reset_index(drop=True)
    df2_temp.drop(['year','mon','day','hour'],axis=1, inplace=True)

    df = pd.concat([df1_temp, df2_temp], axis=1)
    return df


def load_sgp_data(file_dir,file_name):
    fid = Dataset(file_dir+file_name)
    lev_name = [117,138,162,191,226,268,316,374,441,521,607,689,761,819,858,886,912,936,957,976,993]
#feature names
    feats = ["" for x in range(86)]
    feats[0] = 'lhflx'
    feats[1] = 'shflx'
    feats[2] = 'Tsair'
    feats[3] = 'RHair'
    feats[4] = 'cape'
    feats[5] = 'lcl'

    for i in range(6,26):
        feats[i] = "T_"+str(lev_name[i-6])
    for i in range(26,46):
        feats[i] = "Q_"+str(lev_name[i-26])
    for i in range(46,66):
        feats[i] = "divs_"+str(lev_name[i-46])
    for i in range(66,86):
        feats[i] = "divq_"+str(lev_name[i-66])

    #feats[6] = "T_145"
    #feats[15] = "T_600"
    #feats[25] = "T_995"
    #feats[26] = "Q_145"
    #feats[35] = "Q_600"
    #feats[45] = "Q_995"
    #feats[46] = "divs_145"
    #feats[55] = "divs_600"
    #feats[65] = "divs_995"
    #feats[66] = "divq_145"
    #feats[75] = "divq_600"
    #feats[85] = "divq_995"
#load data from netcdf
    nlev = 20

    prect = fid.variables['prect'][:] 
    ntime = len(prect)
    lhflx = fid.variables['lhflx'][:].reshape(ntime,1)
    shflx = fid.variables['shflx'][:].reshape(ntime,1)
    Tsair = fid.variables['Tsair'][:].reshape(ntime,1)
    RHsair = fid.variables['RHsair'][:].reshape(ntime,1)
    T = fid.variables['T'][:,10:].reshape(ntime,nlev)
    q = fid.variables['q'][:,10:].reshape(ntime,nlev)
    divs = fid.variables['divs'][:,10:].reshape(ntime,nlev)
    divq = fid.variables['divq'][:,10:].reshape(ntime,nlev)
    #cape = fid.variables['cape'][:].reshape(ntime,1)
    #lcl = fid.variables['lcl'][:].reshape(ntime,1)
    hour = fid.variables['hour'][:].reshape(ntime,1)
    day  = fid.variables['day'][:].reshape(ntime,1)
    mon  = fid.variables['mon'][:].reshape(ntime,1)
    year = fid.variables['year'][:].reshape(ntime,1)
    lev  = fid.variables['lev'][10:].reshape(nlev,1)


    cape = np.loadtxt(file_dir+"sgp_dilute_dcape").reshape(ntime,1)
    lcl  = np.loadtxt(file_dir+"sgp_dilute_lcl").reshape(ntime,1)

#set the label 
    prec_threshold = 0.5
    label = np.zeros(ntime)
    for i in range(ntime):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime,1)

#packaging data
    trig_dataset = np.concatenate((lhflx,shflx,Tsair,RHsair,cape,lcl,T,q,divs,divq),axis=1)
    trig_dataset_df = pd.DataFrame(trig_dataset, columns=feats)
    trig_dataset_df['label'] = label
    trig_dataset_df['hour']  = hour
    trig_dataset_df['day']  = day
    trig_dataset_df['mon']  = mon
    trig_dataset_df['year']  = year
    trig_dataset_df['PRECT'] = prect.reshape(ntime,1)

    return trig_dataset_df

def load_arm_hy(file_dir,file_name):
    fid = Dataset(file_dir+file_name)

    lev_name = [117,138,162,191,226,268,316,374,441,521,607,689,761,819,858,886,912,936,957,976,993]
#feature names
    feats = ["" for x in range(86)]
    feats[0] = 'lhflx'
    feats[1] = 'shflx'
    feats[2] = 'Tsair'
    feats[3] = 'RHair'
    feats[4] = 'cape'
    feats[5] = 'lcl'

    for i in range(6,26):
        feats[i] = "T_"+str(lev_name[i-6])
    for i in range(26,46):
        feats[i] = "Q_"+str(lev_name[i-26])
    for i in range(46,66):
        feats[i] = "divs_"+str(lev_name[i-46])
    for i in range(66,86):
        feats[i] = "divq_"+str(lev_name[i-66])

#load data from netcdf
    lev = 20

    prect = fid.variables['prect'][:]
    ntime = len(prect)
    lhflx = fid.variables['lhflx'][:].reshape(ntime,1)
    shflx = fid.variables['shflx'][:].reshape(ntime,1)
    Tsair = fid.variables['Tsair'][:].reshape(ntime,1)
    RHsair = fid.variables['RHsair'][:].reshape(ntime,1)
    T = fid.variables['T'][:,10:].reshape(ntime,lev)
    q = fid.variables['q'][:,10:].reshape(ntime,lev)
    divs = fid.variables['divs'][:,10:].reshape(ntime,lev)
    divq = fid.variables['divq'][:,10:].reshape(ntime,lev)
    hour = fid.variables['hour'][:].reshape(ntime,1)
    day  = fid.variables['day'][:].reshape(ntime,1)
    mon  = fid.variables['month'][:].reshape(ntime,1)
    year = fid.variables['year'][:].reshape(ntime,1)
    
    cape = np.loadtxt(file_dir+"/goamazon_dilute_dcape").reshape(ntime,1)
    lcl  = np.loadtxt(file_dir+"goamazon_dilute_lcl").reshape(ntime,1)

#set the label 
    prec_threshold = 0.5
    label = np.zeros(ntime)
    for i in range(ntime):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime,1)

#packaging data
    trig_dataset = np.concatenate((lhflx,shflx,Tsair,RHsair,cape,lcl,T,q,divs,divq),axis=1)
    trig_dataset_df = pd.DataFrame(trig_dataset, columns=feats)
    trig_dataset_df['label'] = label
    trig_dataset_df['hour']  = hour
    trig_dataset_df['mon'] = mon
    trig_dataset_df['day'] = day
    trig_dataset_df['year'] = year
    trig_dataset_df['PRECT'] = prect.reshape(ntime,1)

    return trig_dataset_df

def load_era_obs(fin):
    fid = Dataset(fin)
    lev = 27
 
    feats = ["" for x in range(112)]
    feats = ["" for x in range(85)]
    feats[0] = 'slhf'
    feats[1] = 'sshf'
    feats[2] = 't2m'
    feats[3] = 'd2m'
    for i in range(4,31):
        feats[i] = "t_"+str(i-3)
    for i in range(31,58):
        feats[i] = "q_"+str(i-30)
    for i in range(58,85):
        feats[i] = "r_"+str(i-57)
    #for i in range(85,112):
    #    feats[i] = "w_"+str(i-84)
     
    prect = fid.variables['tp'][:]
    ntime = len(prect)
    slhf = fid.variables['slhf'][:].reshape(ntime,1)
    sshf = fid.variables['sshf'][:].reshape(ntime,1)
    t2m = fid.variables['t2m'][:].reshape(ntime,1)
    d2m = fid.variables['d2m'][:].reshape(ntime,1)
    t   = fid.variables['t'][:,:].reshape(ntime,lev)
    q   = fid.variables['q'][:,:].reshape(ntime,lev)
    r   = fid.variables['r'][:,:].reshape(ntime,lev)
    w   = fid.variables['w'][:,:].reshape(ntime,lev)

    prec_threshold = 0.5
    label = np.zeros(ntime)
    for i in range(ntime):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime,1)

    #trig_dataset = np.concatenate((slhf,sshf,t2m,d2m,t,q,r,w),axis=1)
    trig_dataset = np.concatenate((slhf,sshf,t2m,d2m,t,q,r),axis=1)
    trig_dataset_df = pd.DataFrame(trig_dataset, columns=feats)
    trig_dataset_df['label'] = label

    return trig_dataset_df

def load_goamazon_data():
    fin = "../../data/goamazon/continuous_at_goamazon.nc"
    fid = Dataset(fin)
    lev = 36

    feats = ["" for x in range(150)]
    feats[0] = 'lhflx'
    feats[1] = 'shflx'
    feats[2] = 'Tsair'
    feats[3] = 'RHair'
    feats[4] = 'cape'
    feats[5] = 'lcl'

    for i in range(6,42):
        feats[i] = "T_"+str(i-5)
    for i in range(42,78):
        feats[i] = "Q_"+str(i-41)
    for i in range(78,114):
        feats[i] = "divs_"+str(i-77)
    for i in range(114,150):
        feats[i] = "divq_"+str(i-113)
#    for i in range(150,186):
#        feats[i] = "omega_"+str(i-149)
    
#load data from netcdf
    prect = fid.variables['prec_srf'][:] # mm/h
    ntime = len(prect)
    lhflx = fid.variables['LH'][:].reshape(ntime,1)
    shflx = fid.variables['SH'][:].reshape(ntime,1)
    Tsair = fid.variables['T_srf'][:].reshape(ntime,1)
    T = fid.variables['T'][:,5:41].reshape(ntime,lev)
    q = fid.variables['q'][:,5:41].reshape(ntime,lev) / 1000.0
    omega = fid.variables['omega'][:,5:41].reshape(ntime,lev)
    divs = fid.variables['s_adv_h'][:,5:41].reshape(ntime, lev)
    divq = fid.variables['q_adv_h'][:,5:41].reshape(ntime, lev)
    cape = np.loadtxt("cape_goamazon.out").reshape(ntime,1)
    lcl = np.loadtxt("lcl_goamazon.out").reshape(ntime,1)
    hour = fid.variables['hour'][:].reshape(ntime,1)
    PS = fid.variables['p_srf_aver'][:].reshape(ntime,1)
    q_srf = fid.variables['q_srf'][:].reshape(ntime,1)

# compute the surface relative humidity
    es = 6.112*np.exp(17.67*Tsair/(243.5+Tsair))
    qs = 0.622*es/(PS - es)
    RHsair = q_srf/qs

#set the label
    prec_threshold = 0.5
    label = np.zeros(ntime)
    for i in range(ntime):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime,1)

#packaging data
    trig_dataset = np.concatenate((lhflx,shflx,Tsair,RHsair,cape,lcl,T,q,divs,divq),axis=1)
    trig_dataset_df = pd.DataFrame(trig_dataset, columns=feats)
    trig_dataset_df['label'] = label
    trig_dataset_df['hour']  = hour
    trig_dataset_df['PRECT'] = prect.reshape(ntime,1)
   

    return trig_dataset_df

def load_twp_data():
    fin = "../../data/twp/continuous_at_TWP.nc"
    fid = Dataset(fin)
    lev = 36

    feats = ["" for x in range(150)]
    feats[0] = 'lhflx'
    feats[1] = 'shflx'
    feats[2] = 'Tsair'
    feats[3] = 'RHair'
    feats[4] = 'cape'
    feats[5] = 'lcl'

    for i in range(6,42):
        feats[i] = "T_"+str(i-5)
    for i in range(42,78):
        feats[i] = "Q_"+str(i-41)
    for i in range(78,114):
        feats[i] = "divs_"+str(i-77)
    for i in range(114,150):
        feats[i] = "divq_"+str(i-113)
    
#load data from netcdf
    prect = fid.variables['prec_srf'][:] # mm/h
    ntime = len(prect)
    lhflx = fid.variables['LH'][:].reshape(ntime,1)
    shflx = fid.variables['SH'][:].reshape(ntime,1)
    Tsair = fid.variables['T_srf'][:].reshape(ntime,1) + 273.15
    RHsair = fid.variables['q_srf'][:].reshape(ntime,1)
    T = fid.variables['T'][:,1:37].reshape(ntime,lev)
    q = fid.variables['q'][:,1:37].reshape(ntime,lev) / 1000.0
    divs = fid.variables['s_adv_h'][:,1:37].reshape(ntime, lev)
    divq = fid.variables['q_adv_h'][:,1:37].reshape(ntime, lev)
    cape = np.loadtxt("cape_twp.out").reshape(ntime,1)
    lcl = np.loadtxt("lcl_twp.out").reshape(ntime,1)
    hour = fid.variables['hour'][:].reshape(ntime,1)

#set the label
    prec_threshold = 0.5
    label = np.zeros(ntime)
    for i in range(ntime):
        if(prect[i] >= prec_threshold):
            label[i] = 1
        else:
            label[i] = 0
    label = label.reshape(ntime,1)

#packaging data
    trig_dataset = np.concatenate((lhflx,shflx,Tsair,RHsair,cape,lcl,T,q,divs,divq),axis=1)
    trig_dataset_df = pd.DataFrame(trig_dataset, columns=feats)
    trig_dataset_df['label'] = label
    trig_dataset_df['hour']  = hour
    trig_dataset_df['PRECT'] = prect.reshape(ntime,1)
   
    return trig_dataset_df

if __name__ == '__main__':
    dataset = load_sgp_data("trigger_sgp_hy.nc")
    #a = load_forcing("../../data/goamazon/GOAMAZON_iopfile_4scam.nc")
    #print(a)
