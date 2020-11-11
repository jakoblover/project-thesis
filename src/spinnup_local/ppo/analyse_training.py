import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


root_path = "model_ppo/"
folder = "v1"
path = root_path + folder + "/" + folder + "_s0/progress.txt"
df = pd.read_csv(path, sep="\t")
print(df.head())
print(df.columns)
plt.figure()
plt.title("Average Episode Reward")
plt.xlabel("Epoch")
plt.plot(df["Epoch"],df["AverageEpRet"],label=folder)


folder = "v2"
path = root_path + folder + "/" + folder + "_s0/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df["AverageEpRet"],label =folder)

folder = "v3"
path = root_path + folder + "/" + folder + "_s0/progress.txt"
df = pd.read_csv(path, sep="\t")
plt.plot(df["Epoch"],df["AverageEpRet"],label =folder)

plt.legend()

plt.legend()

plt.show()
