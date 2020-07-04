import pandas as pd 
import matplotlib.pyplot as plt

imp_df = pd.read_csv("imp.txt", sep="\s+")
a= imp_df.set_index('rank')
#imp_df.plot.bar(x='SGP', y='rank')
#plt.show()
print(a)
