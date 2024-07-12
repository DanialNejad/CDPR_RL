import numpy as np
import matplotlib.pyplot as plt

C = [0, 1.15]
r = 0.2
frame_skip = 5
T = np.linspace(0,0.002*frame_skip,10)
phi = np.zeros(len(T)+1)
x = np.zeros(len(T))
y = np.zeros(len(T))
for i in range(len(T)):
    x[i] = C[0] + r*np.sin(phi[i])
    y[i] = C[1] + r*np.cos(phi[i])
    phi[i+1] = phi[i] + 0.7*(1-np.tanh(0.6*T[i]))
    print(f"{x[i]} -0.03 {y[i]}")

plt.figure()
plt.plot(x,y)
plt.axis('equal')

plt.figure()
plt.plot(T,phi[:-1])
plt.show()

