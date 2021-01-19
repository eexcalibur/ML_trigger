import sys
import os

fl = ''
for iy in range(2002, 2010):
    fnames = str(iy)+"/netcdf_data/*0[6-8]01.000000.cdf"
    fl = fl+" "+fnames
print(fl)
os.system("ncrcat -O"+fl+"  Arm_CF_2002_2009.nc")
