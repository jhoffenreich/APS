# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:35:44 2026

@author: hoffe
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 1000
fs = 1000

#%% FUNCIONES
def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn/fs, 1/fs)

    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)

    return tt, xx
#%%
df = fs/N

n = np.arange(N)

# valores de k0
k0_1 = N/4
k0_2 = N/4 + 0.25
k0_3 = N/4 + 0.5

# frecuencias correspondientes
f0_1 = k0_1 * df
f0_2 = k0_2 * df
f0_3 = k0_3 * df

#generamos las 3 senoidales 
A = np.sqrt(2)

x1 = A * np.sin(2*np.pi*f0_1*n/fs)
x2 = A * np.sin(2*np.pi*f0_2*n/fs)
x3 = A * np.sin(2*np.pi*f0_3*n/fs)

#calculamos dft
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X3 = np.fft.fft(x3)

#definimos eje de frecuencias 
f = np.arange(N//2) * df #nos quedamos solo con la mitad postiva, desde 0 a fs/2

#densidad espectral de potencia 
P1 = np.abs(X1/N)**2
P2 = np.abs(X2/N)**2
P3 = np.abs(X3/N)**2
#nos quedamos  con la mitad 
P1 = P1[:N//2]
P2 = P2[:N//2]
P3 = P3[:N//2]

#normalizamos a 0db
P1_dB = 10*np.log10(P1 / np.max(P1))
P2_dB = 10*np.log10(P2 / np.max(P2))
P3_dB = 10*np.log10(P3 / np.max(P3))

#graficamos
plt.figure()

plt.plot(f, P1, label=r'$k_0=N/4$')
plt.plot(f, P2, label=r'$k_0=N/4+0.25$')
plt.plot(f, P3, label=r'$k_0=N/4+0.5$')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PDS normalizada [dB]')
plt.title('Desparramo espectral')
plt.grid()
plt.legend()

plt.xlim(240, 260)#zoom al rededor de las frecuencias con las que trabajamos para ver mejor. 
plt.show()

#agregar grafico que vimos con marti. 

#parte b 
# Potencia en tiempo
Px1_t = np.mean(x1**2)
Px2_t = np.mean(x2**2)
Px3_t = np.mean(x3**2)

# Potencia usando Parseval
Px1_f = np.sum(np.abs(X1)**2) / N**2
Px2_f = np.sum(np.abs(X2)**2) / N**2
Px3_f = np.sum(np.abs(X3)**2) / N**2

print("x1 - potencia en tiempo:", Px1_t)
print("x1 - potencia por Parseval:", Px1_f)

print("x2 - potencia en tiempo:", Px2_t)
print("x2 - potencia por Parseval:", Px2_f)

print("x3 - potencia en tiempo:", Px3_t)
print("x3 - potencia por Parseval:", Px3_f)



#falta explicacion parte b.2 

#c 
# Zero padding
N_pad = 10 * N
# Eje de frecuencia nuevo
f_pad = np.arange(N_pad//2) * fs/N_pad

X1_pad = np.fft.fft(x1, n=N_pad)
X2_pad = np.fft.fft(x2, n=N_pad)
X3_pad = np.fft.fft(x3, n=N_pad)

P1_pad = (2/N**2) * np.abs(X1_pad[:N_pad//2])**2
P2_pad = (2/N**2) * np.abs(X2_pad[:N_pad//2])**2
P3_pad = (2/N**2) * np.abs(X3_pad[:N_pad//2])**2

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)


# --- Señal 1 ---
ax[0].plot(f_pad, P1_pad, label='Con zero padding')
ax[0].stem(f, 2*P1, basefmt='k-', label='Sin zero padding')

ax[0].set_title('k0 = N/4')
ax[0].set_ylabel('Potencia [W]')
ax[0].grid()
ax[0].legend()


# --- Señal 2 ---
ax[1].plot(f_pad, P2_pad, label='Con zero padding')
ax[1].stem(f, 2*P2, basefmt='k-', label='Sin zero padding')

ax[1].set_title('k0 = N/4 + 0.25')
ax[1].set_ylabel('Potencia [W]')
ax[1].grid()
ax[1].legend()


# --- Señal 3 ---
ax[2].plot(f_pad, P3_pad, label='Con zero padding')
ax[2].stem(f, 2*P3, basefmt='k-', label='Sin zero padding')

ax[2].set_title('k0 = N/4 + 0.5')
ax[2].set_ylabel('Potencia [W]')
ax[2].set_xlabel('Frecuencia [Hz]')
ax[2].grid()
ax[2].legend()


plt.xlim(245, 255)
plt.tight_layout()
plt.show()