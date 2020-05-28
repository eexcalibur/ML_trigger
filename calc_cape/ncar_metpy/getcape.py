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

fid = Dataset("/global/homes/z/zhangtao/CPS_trigger/data/goamazon/continuous_at_goamazon.nc")

t = fid.variables['T'][:] #K
print(t.shape)
lev = fid.variables['lev'][:] #mba
print(lev.shape)
q = fid.variables['q'][:] / 1000.0 #g/kg
print(q.shape)

epsilon = 0.622
e = lev * q / (q + epsilon)

Td = (243.5 * np.log(e/6.112)) / (17.67 - np.log(e/6.112)) + 273.15

cape = np.zeros([t.shape[0]])
cin = np.zeros([t.shape[0]])
mx = np.zeros([t.shape[0]])
tmp = np.zeros([t.shape[0]])
print(cape.shape)
print(cape[1])


for i in range(100):
    a,b = getcape.getcape(lev[5:], t[i,5:]-273, Td[i,5:]-273)
    cape[i] = a
    cin[i] = b


np.savetxt("cape_ncar.txt", cape)
