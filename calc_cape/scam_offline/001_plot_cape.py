import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 

#data = np.loadtxt("goamazon_maxi.txt")
data = np.loadtxt("goamazon_cape.txt")
data1 = np.loadtxt("goamazon_dcape.txt")
data2 = np.zeros(data.shape[0])

fig, ax = plt.subplots(1,1,figsize=(7,4))
ax.plot(data / 1000.0, label="CAPE (kJ/kg)")
ax.plot(data1*10 / 1000.0, label="dCAPE (10*kJ/kg/hr)")
ax.axhline(y=0.0, linestyle='-')

ax.set_ylim(-5,10)
#ax[1].plot(data1)

#ax[0].set_title("CAPE")
#ax[1].set_title("dCAPE")

#plt.tight_layout()
ax.legend()
plt.show()

