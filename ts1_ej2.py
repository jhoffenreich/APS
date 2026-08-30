import numpy as np
import matplotlib.pyplot as plt

# Señales
#defino x_c
#x = np.array([1, 1, 1])
#nx = np.array([-1, 0, 1])

#para inciso b
N = 8
n = np.arange(N)
x = (1/2)**n

h = np.array([1, 0, 0, 0, -1])
nh = np.array([0, 1, 2, 3, 4])

# Convolución
y = np.convolve(x, h)
y8=y[:N] #tomamos las primeras N muestras de y[n]
#ny = np.arange(-1,6)#porque comienza en -1+0 y termina en 4+1 y array no incluye el ult num


# Gráfico
plt.figure()
plt.stem(n, y8)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Convolución')
plt.grid()
plt.show()

# DFT de 8 puntos

#y = np.array([1, 1, 0, -1, -1, -1, 0,1])
Y = np.fft.fft(y, N)

print("DFT Y[k]:")
for k in range(N):
    print(f"Y[{k}] = {Y[k]:.4f}")

