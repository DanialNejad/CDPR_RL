import numpy as np
import matplotlib.pyplot as plt

C = [0, 0]
r = 1
T = np.linspace(0,1,10)
phi = np.zeros(len(T)+1)
x = np.zeros(len(T))
y = np.zeros(len(T))
for i in range(len(T)):
    x[i] = C[0] + r*np.sin(phi[i])
    y[i] = C[1] + r*np.cos(phi[i])
    phi[i+1] = phi[i] + 0.9*(1-np.tanh(0.5*T[i]))

plt.figure()
plt.plot(x,y)
plt.axis('equal')

plt.figure()
plt.plot(T,phi[:-1])
plt.show()

print(phi)
