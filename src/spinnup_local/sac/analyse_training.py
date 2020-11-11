import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


root_path = "model/"


folder = "sac_sv2"
path = root_path + folder + "/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df["AverageEpRet"],label =folder)

folder = "sac_sv1"
path = root_path + folder + "/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df["AverageEpRet"],label =folder)

root_path = "../ddpg/model/ddpg_sv1/"
folder = "ddpg_sv1"
path = root_path + folder + "/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df["AverageEpRet"],label =folder)


plt.legend()

plt.show()
