import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 

#data = np.loadtxt("goamazon_maxi.txt")
data = np.loadtxt("goamazon_cape.txt")
#data = np.loadtxt("a")

plt.figure(figsize=(12,4))
plt.plot(data)

plt.show()

