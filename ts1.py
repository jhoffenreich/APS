# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:05:25 2026

@author: hoffe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% DEF.Valores

N = 1000
fs = 20000#[Hz] definimos por inciso 1.1 para obtener 10 muestras/periodo
#%% FUNCIONES

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn/fs, 1/fs)

    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)

    return tt, xx


#%% EJ 1.1
# Senoidal de 2 kHz con al menos 10 muestras por período
ff = 2000#[Hz]
vmax = 1
dc = 0
ph = 0

tt, x1 = mi_funcion_sen(
    vmax=vmax,
    dc=dc,
    ff=ff,
    ph=ph,
    nn=N,
    fs=fs
)

print("Frecuencia de la señal:", ff, "Hz")
print("Frecuencia de muestreo:", fs, "Hz")
print("Muestras por período:", fs/ff)

# GRAFICO SENAL EN EL TIEMPO

plt.figure()
plt.plot(tt[:20], x1[:20], marker='.')#graficamos solo las primeras 20 muestras para visualizar mejor
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Señal senoidal de 2 kHz')
plt.grid()
plt.show()

# FFT

X1 = np.fft.fft(x1)#calc transformada y nos devuelve X1[k] con N valores

delta_f = fs/N #calc. resolucion frecuencial
k = np.arange(N//2)
freq = k * delta_f #frec correspondiente al bin k

mod_X1 = 2*np.abs(X1[:N//2])/N #normalizo por N y mult por 2 para recuperar amplit incicial, por qiedarme con la mitad del espectro. 

plt.figure()
plt.plot(freq, mod_X1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo de la FFT - Senoidal de 2 kHz')
plt.grid()
plt.show()

#%% EJ 1.2
# Misma señal, potencia media 2 W y fase pi/2

vmax2 = 2 #para que la P=2W
ph2 = np.pi/2

tt, x2 = mi_funcion_sen(
    vmax=vmax2,
    dc=0,
    ff=ff,
    ph=ph2,
    nn=N,
    fs=fs
)
#chequeamos con la varianza si la potencia media es correcta, como la media=0, deberian coincidir
var_x2 = np.var(x2) 
print("Varianza de x2:", var_x2, "W")

# GRAFICO SENAL EN EL TIEMPO

plt.figure()
plt.plot(tt[:20], x2[:20], marker='.')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Senoidal de 2 kHz, 2 W y fase pi/2')
plt.grid()
plt.show()

#FFT
X2 = np.fft.fft(x2)

mod_X2 = 2*np.abs(X2[:N//2])/N

plt.figure()
plt.plot(freq, mod_X2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo de la FFT - Senoidal de 2 kHz y 2 W')
plt.grid()
plt.show()

#%% EJ 1.3
# Ruido normalmente distribuido con media = 0 V y varianza = 0.1 W

media = 0
varianza = 0.1
sigma = np.sqrt(varianza)

ruido_gauss = np.random.normal(media,sigma,N)

print("Media medida:", np.mean(ruido_gauss))
print("Varianza medida:", np.var(ruido_gauss))

#GRAF TEMPORAL
plt.figure()
plt.plot(tt, ruido_gauss)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Ruido gaussiano - varianza 0.1 W')
plt.grid()
plt.show()

#FFT
R_gauss = np.fft.fft(ruido_gauss)

mod_R_gauss = 2*np.abs(R_gauss[:N//2])/N  #/N para normalizar y *2 para compensar que solo me quedo con la mitad del espectro

plt.figure()
plt.plot(freq, mod_R_gauss)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo de la FFT - Ruido gaussiano')
plt.grid()
plt.show()

#%% EJ 1.4
# Ruido uniformemente distribuido con media = 0 V y varianza = 0.1 W

varianza = 0.1
A = np.sqrt(3*varianza)

ruido_uniforme = np.random.uniform(-A,A,N) #elige 1000 números al azar entre \(-A\) y \(A\).

print("Media medida:", np.mean(ruido_uniforme))
print("Varianza medida:", np.var(ruido_uniforme))

#graf en t
plt.figure()
plt.plot(tt, ruido_uniforme)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Ruido uniforme - varianza 0.1 W')
plt.grid()
plt.show()

#FFT
R_uniforme = np.fft.fft(ruido_uniforme)

mod_R_uniforme = 2*np.abs(R_uniforme[:N//2])/N

plt.figure()
plt.plot(freq, mod_R_uniforme)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo de la FFT - Ruido uniforme')
plt.grid()
plt.show()

#%% EJ 1.5
# Pulso rectangular de 2 kHz Potencia = 1 W Duty cycle = 50 %

from scipy import signal

x5 = signal.square(2*np.pi*ff*tt, duty=0.5)

pot_x5 = np.mean(x5**2)

print("Potencia medida:", pot_x5, "W")

#grafico en t
plt.figure()
plt.plot(tt[:20], x5[:20], marker='.')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Señal rectangular de 2 kHz - duty 50%')
plt.grid()
plt.show()

#fft
X5 = np.fft.fft(x5)

mod_X5 = 2*np.abs(X5[:N//2])/N

plt.figure()
plt.plot(freq, mod_X5)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo de la FFT - Señal rectangular')
plt.grid()
plt.show()

#%% BONUS - Señal triangular

x_bonus = signal.sawtooth(
    2*np.pi*ff*tt,
    width=0.5
)

plt.figure()
plt.plot(tt[:20], x_bonus[:20], marker='.')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Señal triangular de 2 kHz')
plt.grid()
plt.show()

# FFT
X_bonus = np.fft.fft(x_bonus)

mod_X_bonus = 2*np.abs(X_bonus[:N//2])/N

plt.figure()
plt.plot(freq, mod_X_bonus)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo de la FFT - Señal triangular')
plt.grid()
plt.show()
#%% BONUS - Potencia mediante Parseval

# Potencia calculada en el tiempo
pot_tiempo = np.mean(x2**2)

# FFT completa
X2 = np.fft.fft(x2)

# Potencia calculada mediante Parseval
pot_frec = np.sum(np.abs(X2)**2) / N**2

print("Potencia en el tiempo:", pot_tiempo, "W")
print("Potencia mediante FFT:", pot_frec, "W")