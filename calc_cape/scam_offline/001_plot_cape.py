import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 

#data = np.loadtxt("goamazon_maxi.txt")
data = np.loadtxt("goamazon_cape.txt")
#data1 = np.loadtxt("/global/homes/z/zhangtao/cape/scam/cesm1_2_2/CAPE_new/run/goamazon_dcape.txt")
#diff = data - data1

plt.figure(figsize=(12,4))
#plt.plot(diff[:250])
#print(diff[:250])
plt.plot(data[:250])

plt.show()

