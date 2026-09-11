# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:47:00 2026

@author: hoffe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:35:44 2026

@author: hoffe
"""

import numpy as np
import matplotlib.pyplot as plt


#%% PARÁMETROS Y FUNCIONES

N = 1000
fs = 1000
df = fs / N

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn/fs, 1/fs)

    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)

    return tt, xx


# Valores de k0
k0_1 = N/4
k0_2 = N/4 + 0.25
k0_3 = N/4 + 0.5

# Frecuencias
f0_1 = k0_1 * df
f0_2 = k0_2 * df
f0_3 = k0_3 * df

# Amplitud para potencia unitaria
A = np.sqrt(2)


#%% GENERAMOS LAS 3 SENOIDALES

t, x1 = mi_funcion_sen(vmax=A, dc=0, ff=f0_1, ph=0, nn=N, fs=fs)
_, x2 = mi_funcion_sen(vmax=A, dc=0, ff=f0_2, ph=0, nn=N, fs=fs)
_, x3 = mi_funcion_sen(vmax=A, dc=0, ff=f0_3, ph=0, nn=N, fs=fs)
#Obs: usamos _, porque las 3 senales tienen exactamente el mismo vector de tiempo, 
#entonces nos basta con guardar uno solo, no hace falta t1, t2, t3
#%% GRÁFICO TEMPORAL
plt.figure()

plt.plot(t,x1,color='deepskyblue',
         linestyle='--',linewidth=2,label='250 Hz')

plt.plot(t,x2,':o',color='magenta',label='250.25 Hz')

plt.plot(t,x3,':x',color='limegreen',label='250.5 Hz')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Comparación temporal de las tres senoidales')
plt.xlim(0,0.04)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
#aunque temporalmente las señales parecen muy similares, su DFT puede resultar muy diferente dependiendo de si la frecuencia coincide o no con los bins.
#%% PARTE A - DFT

# Calculamos DFT
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X3 = np.fft.fft(x3)

# Eje de frecuencias
f = np.arange(N//2) * df

# PSD unilat. 
P1 = 2*np.abs(X1[:N//2]/N)**2
P2 = 2*np.abs(X2[:N//2]/N)**2
P3 = 2*np.abs(X3[:N//2]/N)**2
#elevo al cuadrado x potencia y /N para normalizar amp, nos quedamos con la mitad porque la otra es redundante x senal real y multiplicamos por 2 para represen la pot real en esa mitad

#en db
P1_dB = 10*np.log10(P1)
P2_dB = 10*np.log10(P2)
P3_dB = 10*np.log10(P3)


#GRÁFICO PARTE A

plt.figure(figsize=(10,6))

plt.plot(f, P1_dB, label=r'$k_0=N/4$', color='deepskyblue')
plt.plot(f, P2_dB, label=r'$k_0=N/4+0.25$', color='magenta')
plt.plot(f, P3_dB, label=r'$k_0=N/4+0.5$',color='limegreen')


plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Desparramo espectral')
plt.grid()
plt.legend()

plt.xlim(245, 255)

plt.tight_layout()
plt.show()


#%% PARTE B - PARSEVAL

# Potencia en tiempo
Px1_t = np.mean(x1**2)
Px2_t = np.mean(x2**2)
Px3_t = np.mean(x3**2)

# Potencia usando Parseval
Px1_f = np.sum(np.abs(X1)**2) / N**2
Px2_f = np.sum(np.abs(X2)**2) / N**2
Px3_f = np.sum(np.abs(X3)**2) / N**2


print("Señal 1")
print("Potencia en tiempo:", Px1_t)
print("Potencia por Parseval:", Px1_f)
print()

print("Señal 2")
print("Potencia en tiempo:", Px2_t)
print("Potencia por Parseval:", Px2_f)
print()

print("Señal 3")
print("Potencia en tiempo:", Px3_t)
print("Potencia por Parseval:", Px3_f)
print()

#CONLSUION ; VEMOS QUE LA ENERGIA DE ALGO QUE CAE EN UN BIN ENTERO es igual en total a la de algo defasado. 

#explicacion 1.b --> usamos grafico a agregar de sinc con continua y puntos que toma la DFT. 
#%% PARTE B.2 - DTFT CONTINUA Y MUESTRAS DE LA DFT

# Grilla muy fina para visualizar la DTFT
N_dtft = 100*N
f_dtft = np.arange(N_dtft//2)*fs/N_dtft

# FFT muy densa -> aproximación visual de la DTFT
X1_dtft = np.fft.fft(x1,n=N_dtft)[:N_dtft//2]
X2_dtft = np.fft.fft(x2,n=N_dtft)[:N_dtft//2]
X3_dtft = np.fft.fft(x3,n=N_dtft)[:N_dtft//2]

# Potencia de la curva continua
P1_dtft = 2*np.abs(X1_dtft/N)**2
P2_dtft = 2*np.abs(X2_dtft/N)**2
P3_dtft = 2*np.abs(X3_dtft/N)**2

# Potencia de las muestras de la DFT
# IMPORTANTE: misma normalización que la curva continua
P1_dft = 2*np.abs(X1[:N//2]/N)**2
P2_dft = 2*np.abs(X2[:N//2]/N)**2
P3_dft = 2*np.abs(X3[:N//2]/N)**2

fig,ax = plt.subplots(3,1,figsize=(11,9),sharex=True)

ax[0].plot(f_dtft,P1_dtft,color='deepskyblue',linewidth=2,label='DTFT')
ax[0].stem(f,P1_dft,linefmt='b-',markerfmt='bo',basefmt='k-',label='DFT')
ax[0].set_title(r'$k_0=N/4$')
ax[0].set_ylabel('Potencia')
ax[0].grid()
ax[0].legend()

ax[1].plot(f_dtft,P2_dtft,color='limegreen',linewidth=2,label='DTFT')
ax[1].stem(f,P2_dft,linefmt='g-',markerfmt='go',basefmt='k-',label='DFT')
ax[1].set_title(r'$k_0=N/4+0.25$')
ax[1].set_ylabel('Potencia')
ax[1].grid()
ax[1].legend()

ax[2].plot(f_dtft,P3_dtft,color='hotpink',linewidth=2,label='DTFT')
ax[2].stem(f,P3_dft,linefmt='r-',markerfmt='ro',basefmt='k-',label='DFT')
ax[2].set_title(r'$k_0=N/4+0.5$')
ax[2].set_ylabel('Potencia')
ax[2].set_xlabel('Frecuencia [Hz]')
ax[2].grid()
ax[2].legend()

plt.xlim(245,255)
plt.tight_layout()
plt.show()
#%% PARTE C - ZERO PADDING

# Longitud total: N muestras originales + 9*N ceros
N_pad = 10 * N

# Creamos los vectores de zero padding
x1_pad = np.zeros(N_pad)
x2_pad = np.zeros(N_pad)
x3_pad = np.zeros(N_pad)

# Copiamos las señales originales en las primeras N posiciones
x1_pad[:N] = x1
x2_pad[:N] = x2
x3_pad[:N] = x3

# FFT de las señales padeadas
X1_pad = np.fft.fft(x1_pad)
X2_pad = np.fft.fft(x2_pad)
X3_pad = np.fft.fft(x3_pad)

# Resoluciónes frecuenciales
df = fs / N
df_pad = fs / N_pad #con zero padding

print("Delta f sin padding =", df, "Hz")
print("Delta f con padding =", df_pad, "Hz")


# Eje de frecuencias con zero padding
f_pad = np.arange(N_pad//2) * df_pad


# PSD unilateral CON padding
# IMPORTANTE: seguimos dividiendo por N, no por N_pad --> xq seguimos teniendo N muestras reles de la senal
P1_pad = 2*np.abs(X1_pad[:N_pad//2]/N)**2
P2_pad = 2*np.abs(X2_pad[:N_pad//2]/N)**2
P3_pad = 2*np.abs(X3_pad[:N_pad//2]/N)**2

# Pasamos a dB
P1_pad_dB = 10*np.log10(P1_pad)
P2_pad_dB = 10*np.log10(P2_pad)
P3_pad_dB = 10*np.log10(P3_pad)

# PSD
P1_pad = np.abs(X1_pad / N)**2
P2_pad = np.abs(X2_pad / N)**2
P3_pad = np.abs(X3_pad / N)**2

#%% GRÁFICO DE LAS 3 SEÑALES CON ZERO PADDING

plt.figure(figsize=(11,6))

plt.plot(
    f_pad,
    P1_pad_dB,
    label=r'$k_0=N/4$')

plt.plot(f_pad,P2_pad_dB,label=r'$k_0=N/4+0.25$')

plt.plot(f_pad,P3_pad_dB,label=r'$k_0=N/4+0.5$')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD normalizada [dB]')
plt.title('PSD con zero padding')

plt.xlim(245, 255)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#%% COMPARACIÓN CON Y SIN ZERO PADDING

fig,ax = plt.subplots(3,1,figsize=(11,9),sharex=True)

ax[0].plot(f,P1_dB,':x',color='magenta',linewidth=2,label=r'Sin padding, $\Delta f=1$ Hz')
ax[0].plot(f_pad,P1_pad_dB,color='deepskyblue',linewidth=2,label=r'Con padding, $\Delta f=0.1$ Hz')
ax[0].set_title(r'$k_0=N/4$')
ax[0].set_ylabel('Potencia [dB]')
ax[0].grid()
ax[0].legend()

ax[1].plot(f,P2_dB,':x',color='orange',linewidth=2,label=r'Sin padding, $\Delta f=1$ Hz')
ax[1].plot(f_pad,P2_pad_dB,color='limegreen',linewidth=2,label=r'Con padding, $\Delta f=0.1$ Hz')
ax[1].set_title(r'$k_0=N/4+0.25$')
ax[1].set_ylabel('Potencia [dB]')
ax[1].grid()
ax[1].legend()

ax[2].plot(f,P3_dB,':x',color='hotpink',linewidth=2,label=r'Sin padding, $\Delta f=1$ Hz')
ax[2].plot(f_pad,P3_pad_dB,color='dodgerblue',linewidth=2,label=r'Con padding, $\Delta f=0.1$ Hz')
ax[2].set_title(r'$k_0=N/4+0.5$')
ax[2].set_ylabel('Potencia [dB]')
ax[2].set_xlabel('Frecuencia [Hz]')
ax[2].grid()
ax[2].legend()

plt.xlim(240,260)
plt.ylim(-100,5)
plt.tight_layout()
plt.show()