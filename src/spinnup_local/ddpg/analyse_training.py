import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


root_path = "model/"

y_label = "AverageQVals"

folder = "ddpg_sv1/ddpg_sv1"
path = root_path + folder + "/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df[y_label],label =folder)

folder = "ddpg_sv2"
path = root_path + folder + "/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df[y_label],label =folder)


plt.legend()

plt.show()
