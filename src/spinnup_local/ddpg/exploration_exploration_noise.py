
import matplotlib.pyplot as plt
import numpy as np
from ddpg import OUNoise, NormalActionNoise,OrnsteinUhlenbeckActionNoise

print("hei")

noise = [0]*100000
norm = NormalActionNoise(1)
for i in range(len(noise)):
    noise[i] = norm.sample()[0]

plt.figure(1)
plt.plot(noise,label="rand")

plt.figure(2)
plt.hist(noise,label="rand")
plt.legend()

ou = OrnsteinUhlenbeckActionNoise(1)
for i in range(len(noise)):
    noise[i] = ou.sample()[0]

plt.figure(1)
plt.plot(noise,label="ou")
plt.legend()

plt.figure(2)
plt.hist(noise,label="ou")
plt.legend()
plt.show()
