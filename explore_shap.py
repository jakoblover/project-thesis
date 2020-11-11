import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("a_1.csv")
df.drop(df.columns[[0]], axis=1, inplace=True)
df = df.abs()
#df = df.cumsum()
plt.figure()
df.plot()
plt.show()