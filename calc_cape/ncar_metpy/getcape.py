from netCDF4 import Dataset, MFDataset, num2date
import pandas as pd
import numpy as np
import metpy.calc as calc
#from metpy.calc import most_unstable_cape_cin, most_unstable_parcel,pressure_to_height_std,parcel_profile_with_lcl
import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as mpconsts
import matplotlib.pyplot as plt
import getcape
import sys
import matplotlib.pyplot as plt

#fid = Dataset("/global/homes/z/zhangtao/cape/scam/cesm1_2_2/CAPE_new/mymods/Arm_CF_1999_2009_uniform.nc")
fid = Dataset("/global/homes/z/zhangtao/ML_trigger/calc_cape/scam_offline/goamazon_2014_2015.nc")

t = fid.variables['T'][:, ::-1] #K
print(t.shape)
lev = fid.variables['lev'][::-1] #mba
print(lev.shape)
q = fid.variables['q'][:, ::-1] #kg/kg
print(q.shape)

epsilon = 0.622
e = lev * q / (q + epsilon)

Td = (243.5 * np.log(e[:,5:]/6.112)) / (17.67 - np.log(e[:,5:]/6.112)) + 273.15

print(Td.shape)

cape = np.zeros([t.shape[0]])
cin = np.zeros([t.shape[0]])
mx = np.zeros([t.shape[0]])
tmp = np.zeros([t.shape[0]])
print(cape.shape)
print(cape[1])

print(lev[1:].shape)

for i in range(t.shape[0]):
    #a,b = getcape.getcape(lev[5:], t[i,5:]-273, Td[i,5:]-273)
    a,b = getcape.getcape(lev[5:], t[i,5:]-273, Td[i,:]-273)
    cape[i] = a
    cin[i] = b

np.savetxt("cape_ncar.txt", cape)
