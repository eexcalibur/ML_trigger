#****************************************************************
#compute cape
#input: zhang_cps_trigger.nc
#output: cape, lcl
#****************************************************************

# module of CAPE calculation, including saturated vapor pressure calculation, 
#   entropy calculation, and inversion iteration of temperature given entropy
# references: 
# Kerry A. Emannuel, 1994: Atmospheric Convection. Equation 4.3.6, 4.5.9, 
#   4.4.13, 4.4.15
# Raymond and Blythe, 1991, JAS, Equation 1

import numpy as np

#------------------------------------------------------------------------------
# These parameters are from CESM codes

SHR_CONST_BOLTZ   = 1.38065e-23 # Boltzmann's constant ~ J/K/molecule
SHR_CONST_AVOGAD  = 6.02214e26  # Avogadro's number ~ molecules/kmole
SHR_CONST_MWDAIR  = 28.966      # molecular weight dry air ~ kg/kmole
SHR_CONST_MWWV    = 18.016      # molecular weight water vapor ~ kg/kmole
SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ  # Universal gas constant ~ J/K/kmole
Rd   = SHR_CONST_RGAS/SHR_CONST_MWDAIR      # Dry air gas constant     ~ J/K/kg
Rv   = SHR_CONST_RGAS/SHR_CONST_MWWV        # Water vapor gas constant ~ J/K/kg
epsilo = SHR_CONST_MWWV/SHR_CONST_MWDAIR    # ratio of h2o to dry air molecular weights

Tref = 273.15             # reference temperature (C to K conversion)
pref = 1.0e5              # reference pressure (Pa)
Cpd  = 1004.64            # dry air heat capacity at const p (J/kg/K)
Cpv  = 1.810e3            # water vapor heat capacity at const p   (J/kg/K)
Cl   = 4.188e3            # liquid water heat capacity  (J/kg/K)
Lref = 2.501e6            # vaporization heat at Tref (J/kg)
gravity = 9.8 

#------------------------------------------------------------------------------
# calculate the saturated vapor pressure
# T0: temperature in the range [tmin, tmax], unit: K
# flag: 0 --- Emannuel's book, 4.4.13 (for liquid, T>273.15) and 4.4.15 (for ice, T<=273.15)
#       1 --- Bolton (1980) or Emannuel's book, 4.4.14
#       2 --- Use 4.4.13 only (for both liquid and ice)
#       3 --- Use 4.4.15 only (for both liquid and ice)
# es: saturated water vapor pressure (Pa)

def esat(T0, tmin=127.0, tmax=373.0, flag=1):
    T = T0 * 1.0
    if (T < tmin):
        T = tmin
    if (T > tmax):
        T = tmax

    if (flag == 0):
        if (T > Tref):
            es = 100 * np.exp(53.67957 - 6743.769/T - 4.8451*np.log(T))     # Pa
        else:
            es = 100 * np.exp(23.33086 - 6111.72784/T + 0.15215*np.log(T))  # Pa

    if (flag == 1):
        es = 611.2 * np.exp(17.67*(T-Tref) / (T-Tref+243.5))    # Pa

    if (flag == 2):
        es = 100 * np.exp(53.67957 - 6743.769/T - 4.8451*np.log(T))     # Pa

    if (flag == 3):
        es = 100 * np.exp(23.33086 - 6111.72784/T + 0.15215*np.log(T))  # Pa
    
    return es

#------------------------------------------------------------------------------
# calculate entropy (J/kg)
# p:  pressure (Pa)
# T:  temperature (K)
# qt: total water mixing ratio (kg/kg)
# flag: 0 --- Emannuel's book 4.5.9
#       1 --- Raymond and Blythe, 1991, and CAM5 codes

def entropy(p,T, qt, flag=1):
    L  = Lref - (Cl - Cpv)*T    # Latent heat at temperature T (J/kg)
    es = esat(T)                # saturated water vapor pressure (Pa)
    qs = epsilo*es / (p-es)     # saturated water vapor mixing ratio (kg/kg)

    qv = min(qt,qs)             # water vapor mixing ratio    
    e  = qv*p / (qv+epsilo)     # water vapor pressure (Pa)

    if (flag == 0):
        s  = (Cpd + qt*Cl)*np.log(T) - Rd*np.log(p-e) + L*qv/T - qv*Rv*np.log(qv/qs) 

    if (flag == 1):
        s  = (Cpd + qt*Cl)*np.log(T/Tref) - Rd*np.log((p-e)/pref) + L*qv/T - qv*Rv*np.log(qv/qs) 

    return s

#------------------------------------------------------------------------------
# Given entropy (and pressure, total water mixing ratio), calculate temperature 
#   by Newton's iteration method (see CAM5 codes)
# p:  pressure (Pa)
# qt: total water mixing ratio (kg/kg)
# s:  entropy (J/kg)
# flag: the same as function entropy
# loopmax: maximum iteration steps
# dT: calculate ds/dT (K) (the smaller, the faster the iteration)
# pT: precision of iteration on temperature (K)

def ientropy(p, s, qt, loopmax=100, dT=0.001, pT=0.001):

    # Ts: first guess of temperature (K)
    # Based on my tests, the best first guess is the temperature of saturated air
    egs = qt*p/(qt+epsilo)
    x  = np.log(egs/611.2)
    Ts = 243.5*x/(17.67-x) + 273.15

    for i in range(loopmax):
        s1 = entropy(p,Ts,   qt)
        s2 = entropy(p,Ts-dT,qt)
        dsdT = (s1-s2)/dT
        dTs  = (s-s1) / dsdT
        Ts  = Ts + dTs
        if (abs(dTs) < pT):
            return Ts
#    print ' P(mb)= ', p/100.0, ' Tfg(K)= ', T0, ' qt(g/kg) = ', 1000.0*qt
#    print ' qsat(g/kg) = ', 1000.0*epsilo*esat(Ts)/(p-esat(Ts)),', s(J/kg) = ',s
    return Ts

#------------------------------------------------------------------------------
# calculate the density temperature (K), see Emannuel's book equation 4.3.6
# p: pressure (Pa)
# T: temperature (K)
# qt: total water mixing ratio (kg/kg)

def fdenstemp(p, T, qt):
    es = esat(T)                # saturated water vapor pressure (Pa)
    qs = epsilo*es / (p-es)     # saturated water vapor mixing ratio (kg/kg)
    qv = min(qt,qs)             # water vapor mixing ratio    
    Tden = T * (1.0+qv/epsilo) / (1.0+qt)    
    Tv   = T * (1.0+qv/epsilo) / (1.0+qv)
    return Tden, qv, Tv

#------------------------------------------------------------------------------
# calculate density temperature profile of the parcel and environment
# iplev: parcel launching level index

def fprofile(p,T,qt,iplev):
    nlev = np.size(p)
    Tdena = np.zeros([nlev])    # density temperature of environment (K)
    Tva   = np.zeros([nlev])    # virsual temperature of environment (K)
    qva   = np.zeros([nlev])    # water vapor mixing ratio of environment (kg/kg)

    Tdenp = np.zeros([nlev])    # density temperature of parcel      (K)
    Tvp   = np.zeros([nlev])    # virsual temperature of parcel      (K)
    Tp    = np.zeros([nlev])    # temperature of parcel (K)
    qvp   = np.zeros([nlev])    # water vapor mixing ratio of parcel (kg/kg)

    # entropy of parcel
    #sp = entropy(p[iplev],T[iplev],qt[iplev])
    sp = entropy(p[iplev],T[iplev],qt[iplev])
    for ilev in range(nlev):
        Tp[ilev]    = ientropy(p[ilev], sp, qt[iplev])
        Tdena[ilev],qva[ilev],Tva[ilev] = fdenstemp(p[ilev], T[ilev], qt[ilev])
        Tdenp[ilev],qvp[ilev],Tvp[ilev] = fdenstemp(p[ilev], Tp[ilev], qt[iplev])

    return Tdena,Tva,qva,Tdenp,Tvp,Tp,qvp

#------------------------------------------------------------------------------
# calculate the convection available potential energy (CAPE), unit: J/kg
# and   CIN: convection inhibition (J/kg)
#       LCL: lifting condensation level (m): qt >= qs
#       LFC: free convection level (m): Tdenp >= Tdena
#       LNB: level of neutral buoyancy (m): Tdenp <= Tdena
# see Emannuel's book, equation 6.3.5
# launch: 0 --- parcel launch from bottom level
#         1 --- parcel launch from maximum moist static energy (hmax)
#         2 --- parcel launch from surface
# p0: pressure (Pa)
# T0: temperature profile (K)
# qt0: total water mixing ratio (kg/kg)
# z0: geopotential height (m)
# zs: topography (m)

def fcape_1d(p0,T0,qt0,z0,zs, launch=1):
    nlev = np.size(p0)
    ptop = 40000.0 # Pa, limit of launch level
    ctop = 5000.0  # Pa, limit of cloud top

    # order: bottom to top
    if (p0[0] < p0[nlev-1]):
        p  = p0[::-1] * 1.0
        T  = T0[::-1] * 1.0
        qt = qt0[::-1] * 1.0
        z  = z0[::-1] * 1.0
    else:
        p  = p0 * 1.0
        T  = T0 * 1.0
        qt = qt0 * 1.0
        z  = z0 * 1.0

    # different launch methods
    if (launch == 0):
        iplev = 0
    if (launch == 1):
        hmax = 0
        for ilev in range(nlev):
            if (p[ilev] > ptop):
                L  = Lref - (Cl - Cpv)*T[ilev]    # Latent heat at temperature T (J/kg)
                es = esat(T[ilev])                # saturated water vapor pressure (Pa)
                qs = epsilo*es / (p[ilev]-es)     # saturated water vapor mixing ratio (kg/kg)
                qv = min(qt[ilev],qs)             # water vapor mixing ratio    
                h  = (Cpd + qt[ilev]*Cl)*T[ilev] + L*qv + (1+qt[ilev])*gravity*z[ilev]
                if (h > hmax):
                    iplev = ilev
                    hmax = h

    if (launch == 2):
        p2  = np.zeros([nlev+1])
        T2  = np.zeros([nlev+1])
        qt2 = np.zeros([nlev+1])
        z2  = np.zeros([nlev+1])

        if (zs <= z[0]):
            z2[0] = zs
            ratio = (z[0]-zs) / (z[1]-z[0]) 
            p2[0] = np.exp(np.log(p[0]) + (np.log(p[0])- np.log(p[1])) * ratio)
            T2[0] = T[0] + (T[0]- T[1]) * ratio
            qt2[0] = qt[0] + (qt[0]- qt[1]) * ratio

            z2[1:nlev+1] = z
            T2[1:nlev+1] = T
            qt2[1:nlev+1] = qt
            p2[1:nlev+1] = p
            iplev = 0
        else:
            count = 0
            for ilev in range(nlev):
                if z[ilev] < zs:
                    z2[ilev] = z[ilev]
                    T2[ilev] = T[ilev]
                    p2[ilev] = p[ilev]
                    qt2[ilev] = qt[ilev]
                    count = ilev
                else:
                    if (count+1 == ilev):
                        z2[ilev] = zs
                        ratio = (zs-z[ilev]) / (z[ilev+1]-z[ilev])
                        T2[ilev] = T[ilev] - (T[ilev]-T[ilev+1]) * ratio
                        p2[ilev] = p[ilev] - (p[ilev]-p[ilev+1]) * ratio
                        qt2[ilev] = qt[ilev] - (qt[ilev]-qt[ilev+1]) * ratio
                    z2[ilev+1] = z[ilev]
                    T2[ilev+1] = T[ilev]
                    p2[ilev+1] = p[ilev]
                    qt2[ilev+1] = qt[ilev]
            iplev = count + 1
        nlev = nlev + 1
        p = p2 * 1.0
        T = T2 * 1.0
        z = z2 * 1.0
        qt = qt2 * 1.0

    # get parcel buoyancy profile
    Tdena,Tva,qva,Tdenp,Tvp,Tp,qvp = fprofile(p,T,qt,iplev)

    # LCL
    lcl = z[iplev]
    for i in range(iplev+1,nlev):
        if abs(qt[iplev]-qvp[i]) > 1.0e-6:
            lcl = z[i-1]
            break

    # LFC, LNB, CAPE, CIN
    # two types of definition: 
    #   1) difference of density temperature (in Emanuel's book)
    #   2) difference of virsual temperature (in ZM scheme, it is larger than (1) )

#    buoy = Tdenp - Tdena 
    buoy = Tvp - Tva

    buoy[0:iplev+1] = -1.0e-30
    buoy[p<ctop]    = -1.0e-30

    bmax = np.nanmax(buoy)
    if (bmax <= 0):
        cape = 0.0
        cin  = np.nan
        lfc  = np.nan
        lnb  = np.nan
    else:
        dlnp = np.zeros([nlev])
        dlnp[iplev] = (np.log(p[iplev]) - np.log(p[iplev+1]))/2
        for i in range(iplev+1,nlev-1):
            dlnp[i] = (np.log(p[i-1]) - np.log(p[i+1]))/2
        dlnp[p<ctop] = 0.0

        cape = Rd * np.nansum(buoy[buoy>0] * dlnp[buoy>0])  # J/kg

        cin = 0.0
        for i in range(iplev,nlev):
            cin = cin + Rd * buoy[i] * dlnp[i]        # J/kg
            if (buoy[i]<0 and buoy[i+1]>0):
                ratio = -buoy[i]/(buoy[i+1]-buoy[i])
                lfc = z[i] + (z[i+1]-z[i]) * ratio  # m
                break
        
        for i in range(iplev,nlev)[::-1]:
            if (buoy[i]<0 and buoy[i-1]>0):
                ratio = -buoy[i]/(buoy[i-1]-buoy[i])
                lnb = z[i] - (z[i]-z[i-1]) * ratio  # m
                break
        
    return z[iplev],cape,cin,lcl,lfc,lnb
    #return z[iplev],cape,cin,lcl,lfc

#------------------------------------------------------------------------------
# ph: launch parcel height  (m)
# cape: J/kg
# cin: J/kg

def fcape_3d(p,T,qt,z,zs, launch=1):
    ss = T.shape
    nlev = ss[0]
    nlat = ss[1]
    nlon = ss[2]
    ph = np.zeros([nlat,nlon])
    cape = np.zeros([nlat,nlon])
    cin = np.zeros([nlat,nlon])
    lcl = np.zeros([nlat,nlon])
    lfc = np.zeros([nlat,nlon])
    lnb = np.zeros([nlat,nlon])

    print(nlev,nlat,nlon)
    for ilat in range(nlat):
        print (ilat+1)
        
        for ilon in range(nlon):
            ph[ilat,ilon],cape[ilat,ilon],cin[ilat,ilon],lcl[ilat,ilon],lfc[ilat,ilon],lnb[ilat,ilon] = fcape_1d(p[:,ilat,ilon],T[:,ilat,ilon],qt[:,ilat,ilon],z[:,ilat,ilon],zs[ilat,ilon], launch=launch)

    return ph,cape,cin,lcl,lfc,lnb
    
#==============================================================================
#==============================================================================
#==============================================================================
# test codes
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.colors as mc
##import random

#from mod_fplot import fplotcoast, fncread, fplotbox, fsubarray

#pfig = '../figures/'
#ffig = 'test_cape_entropy.png'
#plt.figure(1,figsize=(6,5))
#print pfig+ffig

#p = 102300.0
#T = np.linspace(-40,40,81) + 273.15
#q = np.linspace(1,50,50) * 1e-3

#nT = np.size(T)
#nq = np.size(q)

#qs = np.zeros([nT])
#s  = np.zeros([nq, nT])

#for i in range(nT):
#    es = esat(T[i])
#    qs[i] = epsilo * es / (p-es)
#    for j in range(nq):
#        s[j,i] = entropy(p,T[i],q[j])

#plt.contourf(T-273.15,q*1000, s, 30, cmap = cm.coolwarm)
#plt.colorbar()
#plt.plot(T-273.15,qs*1000,'k')

#plt.grid(True)
#plt.xlabel('T (C)')
#plt.ylabel('qs (g/kg)')

#plt.tight_layout()
#plt.savefig(pfig+ffig)
#plt.close(1)

#------------------------------------------------------------
#iday = 0
##ilat = 40
##ilon = 55
#pin = '../data/'
#T = fncread(pin+'air.1979.nc','air')[iday,:,:,:]  # K
#q = fncread(pin+'q.1979.nc','q')[iday,:,:,:]      # kg/kg
#q = q/(1-q)
#z = fncread(pin+'hgt.1979.nc','hgt')[iday,:,:,:]  # m
#plev = fncread(pin+'hgt.1979.nc','level') * 100.0  # Pa
#p = z * 0.0
#for i in range(np.size(plev)):
#    p[i,:,:] = plev[i]

#zs = fncread(pin+'hgt.sfc.nc','hgt')[0,:,:]     # m
#lat = fncread(pin+'hgt.sfc.nc','lat')
#lon = fncread(pin+'hgt.sfc.nc','lon')

#capencl = fncread(pin+'cape.1979.nc','cape')[iday,:,:]     # m
#cinncl  = -1* fncread(pin+'cape.1979.nc','cin')[iday,:,:]     # m
#lclncl  = fncread(pin+'cape.1979.nc','lcl')[iday,:,:]     # m
#lfcncl  = fncread(pin+'cape.1979.nc','lfc')[iday,:,:]     # m

#capencl[abs(capencl) > 1e10] = np.nan
#cinncl[abs(capencl) > 1e10] = np.nan
#lclncl[abs(capencl) > 1e10] = np.nan
#lfcncl[abs(capencl) > 1e10] = np.nan

##imon = 0
##ilat = 65
##ilon = 106
##pin = '/Users/Oscar/yhy/Work/Data/ERA_Interim/data/years/'
##T = fncread(pin+'ERA_T_1979.nc','t')[imon,8:,ilat,ilon]  # K
##q = fncread(pin+'ERA_Q_1979.nc','q')[imon,8:,ilat,ilon]  # kg/kg
##p = fncread(pin+'ERA_Q_1979.nc','lev')[8:] * 100.0 # hPa
##lat = fncread(pin+'ERA_Q_1979.nc','lat') 
##lon = fncread(pin+'ERA_Q_1979.nc','lon')
##print lat[ilat],lon[ilon] 

#ph,cape,cin,lcl,lfc,lnb = fcape_3d(p,T,q,z,zs,launch=1)

#pfig = '../figures/'
#ffig = 'test_cape_map.png'
#plt.figure(1,figsize=(16,16))
#print pfig+ffig

#alim = np.array([0,360,-90,90])

#for ipic in range(8):
#    if ipic==0:
#        pic = capencl
#        clev = np.linspace(0,2000,21)
#    if ipic==1:
#        pic = cape
#        clev = np.linspace(0,2000,21)
#    if ipic==2:
#        pic = cinncl
#        clev = np.linspace(-400,0,21)
#    if ipic==3:
#        pic = cin
#        clev = np.linspace(-400,0,21)
#    if ipic==4:
#        pic = lclncl
#        clev = np.linspace(0,1000,21)
#    if ipic==5:
#        pic = lcl
#        clev = np.linspace(0,1000,21)
#    if ipic==6:
#        pic = lfcncl
#        clev = np.linspace(0,5000,21)
#    if ipic==7:
#        pic = lfc
#        clev = np.linspace(0,5000,21)

##    pic[zs>100.0] = np.nan

#    plt.subplot(4,2,ipic+1)    
#    plt.contourf(lon,lat,pic, 
#        clev,norm = mc.BoundaryNorm(clev, 256),cmap = cm.coolwarm)
#    plt.colorbar(ticks = clev)

##    plt.contour(lon,lat,cc,[0],colors='k',linewidths=1)

#    fplotcoast(lon)
#    plt.grid(True)
#    plt.xticks(np.linspace(0,360,13))
#    plt.yticks(np.linspace(-90,90,7))
#    plt.axis(alim)

#plt.tight_layout()
#plt.savefig(pfig+ffig)
#plt.close(1)

#----------------------------------------------
#iplev = 0
#Tdena,Tva,qva,Tdenp,Tvp,Tp,qvp = fprofile(p,T,q,iplev=iplev)

#pfig = '../figures/'
#ffig = 'test_cape_profile.png'
#plt.figure(1,figsize=(10,12))
#print pfig+ffig

#plt.subplot(2,2,1)
#plt.plot(T,p/100,'g-')
#plt.plot(Tp,p/100,'r-o')
#plt.plot(Tdena,p/100,'b--')
#plt.plot(Tdenp,p/100,'m--o')
##plt.yscale('log')
#plt.xlabel('Temperature (K)')
#plt.ylabel('pressure (hPa)')
#plt.yticks(np.linspace(0,1100,12))
#plt.grid(True)
#plt.axis([100,305,0,1020])
#plt.gca().invert_yaxis()

#plt.subplot(2,2,3)
##plt.plot(T,p/100,'g-')
#plt.plot(Tp-T,p/100,'r-o')
#plt.plot(Tdenp-Tdena,p/100,'b--')
#plt.plot(Tvp-Tva,p/100,'m--o')
##plt.yscale('log')
#plt.xlabel('Temperature (K)')
#plt.ylabel('pressure (hPa)')
#plt.yticks(np.linspace(0,1100,12))
#plt.grid(True)
#plt.axis([-5,5,0,1020])
#plt.gca().invert_yaxis()

#plt.subplot(2,2,2)
#plt.plot(q*1000,p/100,'g-o')
#plt.plot(q[iplev]*1000+p*0,p/100,'r-o')
#plt.plot(qva*1000,p/100,'b--o')
#plt.plot(qvp*1000,p/100,'m--o')
##plt.yscale('log')
#plt.xlabel('mixing ratio (g/kg)')
#plt.ylabel('pressure (hPa)')
#plt.yticks(np.linspace(0,1100,12))
#plt.grid(True)
#plt.axis([0,20,0,1020])
#plt.gca().invert_yaxis()

#plt.tight_layout()
#plt.savefig(pfig+ffig)
#plt.close(1)


#---------------------------------

#p = 50000.0
#T = 50.0 + 273.15
#q = 50e-3
#s = entropy(p,T,q,flag=1)
#Ts,n = ientropy(p,s,q, 
#            flag=1, dT=1.0, weight=1)
#print s,n,Ts,Ts-T

#---------------------------------
#pfig = '../figures/'
#ffig = 'test_cape_ientropy.png'
#plt.figure(1,figsize=(14,4))
#print pfig+ffig

#p = 102300.0
#T = np.linspace(-70,70,101) + 273.15
#q = np.linspace(1,500,100) * 1e-3

#nT = np.size(T)
#nq = np.size(q)

#qs = np.zeros([nT])
#s  = np.zeros([nq, nT])
#Ts = np.zeros([nq, nT])
#ns = np.zeros([nq, nT])

#for i in range(nT):
#    es = esat(T[i])
#    qs[i] = epsilo * es / (p-es)
#    for j in range(nq):
#        s[j,i] = entropy(p,T[i],q[j],flag=1)
#        Ts[j,i],ns[j,i] = ientropy(p,s[j,i],q[j],dT=0.001)
## random.uniform(-30,30)+T[i]
#        Ts[j,i] = Ts[j,i] - T[i]

#plt.subplot(1,3,1)
#plt.contourf(T-273.15,q*1000, s, 30, cmap = cm.coolwarm)
#plt.colorbar()
#plt.plot(T-273.15,qs*1000,'k')
#plt.grid(True)
#plt.xlabel('T (C)')
#plt.ylabel('q (g/kg)')
#plt.title('entropy (J/kg)')

#clevpt = np.linspace(-0.001,0.001,41)
#plt.subplot(1,3,2)
#plt.contourf(T-273.15,q*1000, Ts, 
#    clevpt,norm = mc.BoundaryNorm(clevpt, 256), cmap = cm.coolwarm)
#plt.colorbar()
##plt.contour(T-273.15,q*1000, Ts, [0], colors='k')
#plt.plot(T-273.15,qs*1000,'k')
#plt.grid(True)
#plt.xlabel('T (C)')
#plt.ylabel('q (g/kg)')
#plt.title('precision (K), min='+format(np.min(Ts),'.4f')+', max='+format(np.max(Ts),'.4f'))

#plt.subplot(1,3,3)
#plt.contourf(T-273.15,q*1000, ns, 30, cmap = cm.coolwarm)
#plt.colorbar()
#plt.plot(T-273.15,qs*1000,'k')
#plt.grid(True)
#plt.xlabel('T (C)')
#plt.ylabel('q (g/kg)')
#plt.title('iteration number, mean='+format(np.mean(ns)))

#plt.tight_layout()
#plt.savefig(pfig+ffig)
#plt.close(1)
#--------------------------------------
#p = 102300.0
#T = 273.15
#qt = 10e-3
#es = esat(T)
#qs = epsilo*es / (p-es)
#qv = min(qs,qt)

#print entropy(p,T,qt), es, qs, qv

#------------------------------------
#pfig = '../figures/'
#ffig = 'test_cape_entropy.png'
#plt.figure(1,figsize=(6,5))
#print pfig+ffig

#p = 102300.0
#T = np.linspace(-40,40,81) + 273.15
#q = np.linspace(1,50,50) * 1e-3

#nT = np.size(T)
#nq = np.size(q)

#qs = np.zeros([nT])
#s  = np.zeros([nq, nT])

#for i in range(nT):
#    es = esat(T[i])
#    qs[i] = epsilo * es / (p-es)
#    for j in range(nq):
#        s[j,i] = entropy(p,T[i],q[j])

#plt.contourf(T-273.15,q*1000, s, 30, cmap = cm.coolwarm)
#plt.colorbar()
#plt.plot(T-273.15,qs*1000,'k')

#plt.grid(True)
#plt.xlabel('T (C)')
#plt.ylabel('qs (g/kg)')

#plt.tight_layout()
#plt.savefig(pfig+ffig)
#plt.close(1)

#-------------------------------------------

from netCDF4 import Dataset
import sys

if __name__ == '__main__':
    din = "../../data/goamazon/"
    din = "../../data/sgp/ContForcing_4scam/"
    #din = "../data/IOPForcing_4scam/"
    #fin = din + "IOP_4scam_sgp9706.nc"
    fin = din + "trigger_1999_2008_summer.nc"
    fid = Dataset(fin)
    
    p  = fid.variables['lev'][5:41] * 100.0
    t  = fid.variables['T'][:,5:41]
    q  = fid.variables['q'][:,5:41] / 1000.0
    ps = fid.variables['PS'][:]
    
    #p  = fid.variables['lev'][1:37] * 100.0
    #t  = fid.variables['T'][:,1:37]
    #q  = fid.variables['q'][:,1:37] / 1000.0
    #ps = fid.variables['p_srf_aver'][:]
    
    p = p[::-1]
    t = t[:,::-1]
    q = q[:,::-1]
    

    print(p)

    #p  = fid.variables['lev'][0:34]
    #t  = fid.variables['T'][:,0:34,0,0]
    #q  = fid.variables['q'][:,0:34,0,0]
    #ps = fid.variables['Ps'][:,0,0]/100
    
    z  = 0.3048*(1 - (p[:]/100/1013.25)**0.190284)*145366.45
    zs = 0.3048*(1 - (ps/1013.25)**0.190284)*145366.45
    #pr = fid.variables['Prec'][:,0,0] * 3600
    
    
    print(p.shape)
    print(t.shape)
    print(q.shape)
    print(z.shape)
    print(ps.shape)
    print(zs.shape)
    #print(pr.shape)
    
    
    ntime = len(ps)
    print(ntime)
    ph    = np.zeros(ntime)
    cape  = np.zeros(ntime)
    cin   = np.zeros(ntime)
    lcl   = np.zeros(ntime)
    lfc   = np.zeros(ntime)
    lnb   = np.zeros(ntime)
    
    
    for itime in range(len(ps)):
    	ph[itime], cape[itime], cin[itime], lcl[itime], lfc[itime], lnb[itime] = fcape_1d(p[:],t[itime,:], q[itime,:], z[:], zs[itime], 1)
    	#ph[itime], cape[itime], cin[itime], lcl[itime], lfc[itime] = fcape_1d(p[:],t[itime,:], q[itime,:], z[:], zs[itime], 1)
    
    lcl_pa = (1 - lcl/0.3048/145366.45) ** (1/0.190284) * 100 * 1013.25
    
    np.savetxt('cape.out',cape)
    np.savetxt('lcl.out',lcl_pa, fmt="%.0f")
    #np.savetxt('pr.out', pr, fmt="%.3f")
