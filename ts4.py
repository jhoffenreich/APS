# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:22:50 2026

@author: hoffe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')

#%% PARAMETROS

N = 1000#muestras temps          
R = 200#realizaciones          
#tendremos 200 senales con 1000 muestras cada una
a0 = np.sqrt(2)   #amp senoidal -> potencia unitaria

omega0 = np.pi/2  #frec angular nominal
k0 = N//4 # omega0 = pi/2 corresponde al bin N/4
SNR_dB = 10  

#%% FRECUENCIAS ALEATORIAS DE LAS 200 REALIZACIONES
fr = np.random.uniform(-2, 2, R)#generamos los 200 valores aleatorio de fr  
#hasta ahora tiene forma: [200] vectir de 200 elementos, con nums entre -2 y +2
omega1 = omega0 + fr*(2*np.pi/N)#calculamos la frec ang real de cada realiz, obtenemos 200 valores dist
#tamb es un vector de 200 valores: [200] y como cada senoidal es una columna
#lo convierto en (1,R) usando RESHAPE  para despue spoder multiplicarlo por n que es de (N,1) tiempo
#entonces al multiplicar omega1.n obtenemos una matriz NxR donde cada sneoidal tiene una frec disnt
omega1 = omega1.reshape(1, R)#ahora (1,200)
n = np.arange(N).reshape(N, 1)#vector columna (1000,1)

#%% GENERO SENOIDALES
s = a0 * np.sin(n * omega1)#al multiplicar n*omega 1 --> broadcasting --> me genera una matriz N*M
#s --> {1000,200}, cada COLUM ES UNA SENAL DISTINTA
#s[:, 0] toda la primer senal
#s[:, 1] toda la segunda
#s[:, 199] la ultima

#%% RUIDO SEGUN SNR

Ps = 1

SNR1 = 3
SNR2 = 10

# -------------------------
# SNR = 3 dB
# -------------------------

sigma2_3 = Ps / (10**(SNR1/10))
sigma_3 = np.sqrt(sigma2_3)

#generamos una matriz de ruido de N*R [1000,200]
na_3 = np.random.normal(0, sigma_3, (N, R))
#cada columna tiene un ruido gausiano distinto


# -------------------------
# SNR = 10 dB
# -------------------------

sigma2_10 = Ps / (10**(SNR2/10))
sigma_10 = np.sqrt(sigma2_10)

#generamos una matriz de ruido de N*R [1000,200]
na_10 = np.random.normal(0, sigma_10, (N, R))
#cada columna tiene un ruido gausiano distinto


#%% SEÑAL FINAL

# usamos LAS MISMAS 200 senoidales para los dos SNR
# entonces la unica diferencia entre los experimentos es el nivel de ruido

x_3 = s + na_3       #tamano [1000,200]
x_10 = s + na_10     #tamano [1000,200]


#%% VENTANAS

#multiplicaremos por las 200 senoidales, queremos que sea una columna de tamano N --> [1000,1]
#despues cuando hacemos multiplicacion numpy x broadcasting replica x 200 realizaciones y logra multiplicar la ventana a las 200 x
#ENTONCES LAS GENERAMOS COMO UN VECTOR DE N ELEMENTOS

w_rect = signal.windows.boxcar(N)
w_flat = signal.windows.flattop(N)
w_black = signal.windows.blackmanharris(N)
w_ham = signal.windows.hamming(N)

# Las convertimos en vectores columna (N,1)
# para poder multiplicarlas por x, que tiene tamaño (N,R)

w_rect = w_rect.reshape(N, 1)
w_flat = w_flat.reshape(N, 1)
w_black = w_black.reshape(N, 1)
w_ham = w_ham.reshape(N, 1)


#%% APLICAMOS LAS VENTANAS

#tamano devuelto: [1000,200]

# -------------------------
# SNR = 3 dB
# -------------------------

x3_rect = x_3 * w_rect
x3_flat = x_3 * w_flat
x3_black = x_3 * w_black
x3_ham = x_3 * w_ham


# -------------------------
# SNR = 10 dB
# -------------------------

x10_rect = x_10 * w_rect
x10_flat = x_10 * w_flat
x10_black = x_10 * w_black
x10_ham = x_10 * w_ham


#%% FFT DE CADA SENAL VENTANEADA

#tenemos que transformar en el tiempo, y este se representa x columna entonces usamos axis=0
#axis=0 transforma sobre la columna
#axis=1 transforma sobre la fila

# -------------------------
# SNR = 3 dB
# -------------------------

fft3_rect = np.fft.fft(x3_rect, axis=0)
fft3_flat = np.fft.fft(x3_flat, axis=0)
fft3_black = np.fft.fft(x3_black, axis=0)
fft3_ham = np.fft.fft(x3_ham, axis=0)


# -------------------------
# SNR = 10 dB
# -------------------------

fft10_rect = np.fft.fft(x10_rect, axis=0)
fft10_flat = np.fft.fft(x10_flat, axis=0)
fft10_black = np.fft.fft(x10_black, axis=0)
fft10_ham = np.fft.fft(x10_ham, axis=0)

#seguimos con tamano [1000,200]
#TRANSFORMAMOS CADA COLUMNA DEL DOM TEMPORAL AL FRECUENCIAL


#COMO X ES REAL, EL ESPECTRO ES CONJUGADO SIMETRICO
#NOS QUEDAMOS CON LA MITAD POSITIVA

# SNR = 3 dB
mod3_rect = np.abs(fft3_rect[:N//2, :])
mod3_flat = np.abs(fft3_flat[:N//2, :])
mod3_black = np.abs(fft3_black[:N//2, :])
mod3_ham = np.abs(fft3_ham[:N//2, :])

# SNR = 10 dB
mod10_rect = np.abs(fft10_rect[:N//2, :])
mod10_flat = np.abs(fft10_flat[:N//2, :])
mod10_black = np.abs(fft10_black[:N//2, :])
mod10_ham = np.abs(fft10_ham[:N//2, :])

#ahora tamano [500,200]


#%% COMPARACION DE VENTANAS EN dB - UNA REALIZACION

i = 0
k = np.arange(N//2)

# Función para pasar el espectro a dB
def espectro_db(X):

    # Seleccionamos una realizacion y normalizamos respecto de su maximo
    X_norm = X[:, i] / np.max(X[:, i])

    # Pasamos a dB
    # 1e-12 evita calcular log10(0)
    return 20 * np.log10(X_norm + 1e-12)


plt.figure(figsize=(12,8))


# -------------------------
# SNR = 3 dB
# -------------------------

plt.subplot(2,1,1)

plt.plot(k, espectro_db(mod3_rect), label='Rectangular')
plt.plot(k, espectro_db(mod3_flat), label='Flat-top')
plt.plot(k, espectro_db(mod3_black), label='Blackman-Harris')
plt.plot(k, espectro_db(mod3_ham), label='Hamming')

plt.axvline(k0, color='black', linestyle='--',
            linewidth=1.5, label=r'$N/4$')

plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.title('Comparación de ventanas - SNR = 3 dB')

plt.xlim(k0-50, k0+50)
plt.ylim(-100, 5)

plt.grid()
plt.legend()


# -------------------------
# SNR = 10 dB
# -------------------------

plt.subplot(2,1,2)

plt.plot(k, espectro_db(mod10_rect), label='Rectangular')
plt.plot(k, espectro_db(mod10_flat), label='Flat-top')
plt.plot(k, espectro_db(mod10_black), label='Blackman-Harris')
plt.plot(k, espectro_db(mod10_ham), label='Hamming')

plt.axvline(k0, color='black', linestyle='--',
            linewidth=1.5, label=r'$N/4$')

plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.title('Comparación de ventanas - SNR = 10 dB')

plt.xlim(k0-50, k0+50)
plt.ylim(-100, 5)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


#%% ESTIMADORES DE AMPLITUD

# Tomamos el bin k0 = N/4 para las R realizaciones
# Como las realizaciones están en columnas, tomamos solo:
# fila k0, todas las columnas --> nos queda vector [200]
#normalizamos todas de la misma manera con 2/N


# -------------------------
# SNR = 3 dB
# -------------------------

est_a3_rect = (2/N) * mod3_rect[k0, :]
est_a3_flat = (2/N) * mod3_flat[k0, :]
est_a3_black = (2/N) * mod3_black[k0, :]
est_a3_ham = (2/N) * mod3_ham[k0, :]


# -------------------------
# SNR = 10 dB
# -------------------------

est_a10_rect = (2/N) * mod10_rect[k0, :]
est_a10_flat = (2/N) * mod10_flat[k0, :]
est_a10_black = (2/N) * mod10_black[k0, :]
est_a10_ham = (2/N) * mod10_ham[k0, :]


#%% ESTIMADORES DE FRECUENCIA

# Buscamos en qué bin está el máximo de cada realización
# axis=0 porque recorremos verticalmente los bins de frecuencia


# -------------------------
# SNR = 3 dB
# -------------------------

est_f3_rect = np.argmax(mod3_rect, axis=0)
est_f3_flat = np.argmax(mod3_flat, axis=0)
est_f3_black = np.argmax(mod3_black, axis=0)
est_f3_ham = np.argmax(mod3_ham, axis=0)


# -------------------------
# SNR = 10 dB
# -------------------------

est_f10_rect = np.argmax(mod10_rect, axis=0)
est_f10_flat = np.argmax(mod10_flat, axis=0)
est_f10_black = np.argmax(mod10_black, axis=0)
est_f10_ham = np.argmax(mod10_ham, axis=0)


#%% HISTOGRAMAS - ESTIMADORES ORIGINALES

bins_hist = 15
transp = 0.5

plt.figure(figsize=(14,10))


# -------------------------
# AMPLITUD - SNR = 3 dB
# -------------------------

plt.subplot(2,2,1)

plt.hist(est_a3_rect, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_a3_ham, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_a3_black, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_a3_flat, bins=bins_hist, alpha=transp, label='Flat-top')

# Verdadero valor de amplitud
plt.axvline(a0, color='black', linestyle='--',
            linewidth=2, label=fr'$a_0={a0:.3f}$')

plt.title('Estimación de amplitud - SNR = 3 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# AMPLITUD - SNR = 10 dB
# -------------------------

plt.subplot(2,2,2)

plt.hist(est_a10_rect, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_a10_ham, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_a10_black, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_a10_flat, bins=bins_hist, alpha=transp, label='Flat-top')

plt.axvline(a0, color='black', linestyle='--',
            linewidth=2, label=fr'$a_0={a0:.3f}$')

plt.title('Estimación de amplitud - SNR = 10 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA - SNR = 3 dB
# -------------------------

plt.subplot(2,2,3)

plt.hist(est_f3_rect, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_f3_ham, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_f3_black, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_f3_flat, bins=bins_hist, alpha=transp, label='Flat-top')

# k0 es el CENTRO de las frecuencias posibles
# no la frecuencia verdadera de cada realización
plt.axvline(k0, color='black', linestyle='--',
            linewidth=2, label=r'$k_0=N/4$')

plt.title('Estimación de frecuencia - SNR = 3 dB')
plt.xlabel('Bin estimado')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA - SNR = 10 dB
# -------------------------

plt.subplot(2,2,4)

plt.hist(est_f10_rect, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_f10_ham, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_f10_black, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_f10_flat, bins=bins_hist, alpha=transp, label='Flat-top')

plt.axvline(k0, color='black', linestyle='--',
            linewidth=2, label=r'$k_0=N/4$')

plt.title('Estimación de frecuencia - SNR = 10 dB')
plt.xlabel('Bin estimado')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


#%% ESTADISTICAS

# frecuencia verdadera de cada realizacion expresada en bins
# como fr esta entre -2 y +2, f_real esta entre 248 y 252
f_real = k0 + fr


# =========================================================
# SNR = 3 dB
# =========================================================

# AMPLITUD

media_a3_rect = np.mean(est_a3_rect)
media_a3_flat = np.mean(est_a3_flat)
media_a3_black = np.mean(est_a3_black)
media_a3_ham = np.mean(est_a3_ham)

sesgo_a3_rect = media_a3_rect - a0
sesgo_a3_flat = media_a3_flat - a0
sesgo_a3_black = media_a3_black - a0
sesgo_a3_ham = media_a3_ham - a0

var_a3_rect = np.var(est_a3_rect)
var_a3_flat = np.var(est_a3_flat)
var_a3_black = np.var(est_a3_black)
var_a3_ham = np.var(est_a3_ham)


# FRECUENCIA

# los estimadores est_f ya estan expresados en bins
# es decir, en multiplos de Delta f

media_f3_rect = np.mean(est_f3_rect)
media_f3_flat = np.mean(est_f3_flat)
media_f3_black = np.mean(est_f3_black)
media_f3_ham = np.mean(est_f3_ham)

# Sesgo = media estimada - media verdadera
sesgo_f3_rect = media_f3_rect - np.mean(f_real)
sesgo_f3_flat = media_f3_flat - np.mean(f_real)
sesgo_f3_black = media_f3_black - np.mean(f_real)
sesgo_f3_ham = media_f3_ham - np.mean(f_real)

# Varianza del estimador de frecuencia
var_f3_rect = np.var(est_f3_rect)
var_f3_flat = np.var(est_f3_flat)
var_f3_black = np.var(est_f3_black)
var_f3_ham = np.var(est_f3_ham)


# =========================================================
# SNR = 10 dB
# =========================================================

# AMPLITUD

media_a10_rect = np.mean(est_a10_rect)
media_a10_flat = np.mean(est_a10_flat)
media_a10_black = np.mean(est_a10_black)
media_a10_ham = np.mean(est_a10_ham)

sesgo_a10_rect = media_a10_rect - a0
sesgo_a10_flat = media_a10_flat - a0
sesgo_a10_black = media_a10_black - a0
sesgo_a10_ham = media_a10_ham - a0

var_a10_rect = np.var(est_a10_rect)
var_a10_flat = np.var(est_a10_flat)
var_a10_black = np.var(est_a10_black)
var_a10_ham = np.var(est_a10_ham)


# FRECUENCIA

media_f10_rect = np.mean(est_f10_rect)
media_f10_flat = np.mean(est_f10_flat)
media_f10_black = np.mean(est_f10_black)
media_f10_ham = np.mean(est_f10_ham)

# Sesgo = media estimada - media verdadera
sesgo_f10_rect = media_f10_rect - np.mean(f_real)
sesgo_f10_flat = media_f10_flat - np.mean(f_real)
sesgo_f10_black = media_f10_black - np.mean(f_real)
sesgo_f10_ham = media_f10_ham - np.mean(f_real)

# Varianza del estimador
var_f10_rect = np.var(est_f10_rect)
var_f10_flat = np.var(est_f10_flat)
var_f10_black = np.var(est_f10_black)
var_f10_ham = np.var(est_f10_ham)
#%% TABLAS DE RESULTADOS

def mostrar_tabla(snr,
                  media_ar, sesgo_ar, var_ar,
                  media_af, sesgo_af, var_af,
                  media_ab, sesgo_ab, var_ab,
                  media_ah, sesgo_ah, var_ah,
                  media_fr, sesgo_fr, var_fr,
                  media_ff, sesgo_ff, var_ff,
                  media_fb, sesgo_fb, var_fb,
                  media_fh, sesgo_fh, var_fh):

    print(f"\n{'='*70}")
    print(f"RESULTADOS - SNR = {snr} dB")
    print(f"{'='*70}")

    # -------------------------
    # TABLA DE AMPLITUD
    # -------------------------

    print(f"\n{'ESTIMACION DE AMPLITUD':^70}")
    print('-'*70)
    print(f"{'Ventana':<20} {'Media':>15} {'Sesgo':>15} {'Varianza':>15}")
    print('-'*70)

    print(f"{'Rectangular':<20} {media_ar:>15.6f} {sesgo_ar:>15.6f} {var_ar:>15.6f}")
    print(f"{'Flat-top':<20} {media_af:>15.6f} {sesgo_af:>15.6f} {var_af:>15.6f}")
    print(f"{'Blackman-Harris':<20} {media_ab:>15.6f} {sesgo_ab:>15.6f} {var_ab:>15.6f}")
    print(f"{'Hamming':<20} {media_ah:>15.6f} {sesgo_ah:>15.6f} {var_ah:>15.6f}")

    # -------------------------
    # TABLA DE FRECUENCIA
    # -------------------------

    print(f"\n{'ESTIMACION DE FRECUENCIA':^70}")
    print('-'*70)
    print(f"{'Ventana':<20} {'Media':>15} {'Sesgo':>15} {'Varianza':>15}")
    print('-'*70)

    print(f"{'Rectangular':<20} {media_fr:>15.6f} {sesgo_fr:>15.6f} {var_fr:>15.6f}")
    print(f"{'Flat-top':<20} {media_ff:>15.6f} {sesgo_ff:>15.6f} {var_ff:>15.6f}")
    print(f"{'Blackman-Harris':<20} {media_fb:>15.6f} {sesgo_fb:>15.6f} {var_fb:>15.6f}")
    print(f"{'Hamming':<20} {media_fh:>15.6f} {sesgo_fh:>15.6f} {var_fh:>15.6f}")


# Tabla SNR = 3 dB
mostrar_tabla(
    3,
    media_a3_rect, sesgo_a3_rect, var_a3_rect,
    media_a3_flat, sesgo_a3_flat, var_a3_flat,
    media_a3_black, sesgo_a3_black, var_a3_black,
    media_a3_ham, sesgo_a3_ham, var_a3_ham,
    media_f3_rect, sesgo_f3_rect, var_f3_rect,
    media_f3_flat, sesgo_f3_flat, var_f3_flat,
    media_f3_black, sesgo_f3_black, var_f3_black,
    media_f3_ham, sesgo_f3_ham, var_f3_ham
)

# Tabla SNR = 10 dB
mostrar_tabla(
    10,
    media_a10_rect, sesgo_a10_rect, var_a10_rect,
    media_a10_flat, sesgo_a10_flat, var_a10_flat,
    media_a10_black, sesgo_a10_black, var_a10_black,
    media_a10_ham, sesgo_a10_ham, var_a10_ham,
    media_f10_rect, sesgo_f10_rect, var_f10_rect,
    media_f10_flat, sesgo_f10_flat, var_f10_flat,
    media_f10_black, sesgo_f10_black, var_f10_black,
    media_f10_ham, sesgo_f10_ham, var_f10_ham
)






#%% CALIBRACION - ESTIMADORES INSESGADOS

#modificamos sesgo no varianza

# -------------------------
# SNR = 3 dB
# -------------------------

est_a3_rect_cal = est_a3_rect - sesgo_a3_rect
est_a3_flat_cal = est_a3_flat - sesgo_a3_flat
est_a3_black_cal = est_a3_black - sesgo_a3_black
est_a3_ham_cal = est_a3_ham - sesgo_a3_ham
est_f3_rect_cal = est_f3_rect - sesgo_f3_rect
est_f3_flat_cal = est_f3_flat - sesgo_f3_flat
est_f3_black_cal = est_f3_black - sesgo_f3_black
est_f3_ham_cal = est_f3_ham - sesgo_f3_ham


# -------------------------
# SNR = 10 dB
# -------------------------

est_a10_rect_cal = est_a10_rect - sesgo_a10_rect
est_a10_flat_cal = est_a10_flat - sesgo_a10_flat
est_a10_black_cal = est_a10_black - sesgo_a10_black
est_a10_ham_cal = est_a10_ham - sesgo_a10_ham
est_f10_rect_cal = est_f10_rect - sesgo_f10_rect
est_f10_flat_cal = est_f10_flat - sesgo_f10_flat
est_f10_black_cal = est_f10_black - sesgo_f10_black
est_f10_ham_cal = est_f10_ham - sesgo_f10_ham


#%% HISTOGRAMAS CALIBRADOS
#ahora las evaluamos solo x varianza, se supone que ya no hay sesgo

plt.figure(figsize=(14,10))


# -------------------------
# AMPLITUD - SNR = 3 dB
# -------------------------

plt.subplot(2,2,1)

plt.hist(est_a3_rect_cal, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_a3_ham_cal, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_a3_black_cal, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_a3_flat_cal, bins=bins_hist, alpha=transp, label='Flat-top')

plt.axvline(a0, color='black', linestyle='--',
            linewidth=2, label=fr'$a_0={a0:.3f}$')

plt.title('Amplitud calibrada - SNR = 3 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# AMPLITUD - SNR = 10 dB
# -------------------------

plt.subplot(2,2,2)

plt.hist(est_a10_rect_cal, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_a10_ham_cal, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_a10_black_cal, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_a10_flat_cal, bins=bins_hist, alpha=transp, label='Flat-top')

plt.axvline(a0, color='black', linestyle='--',
            linewidth=2, label=fr'$a_0={a0:.3f}$')

plt.title('Amplitud calibrada - SNR = 10 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA - SNR = 3 dB
# -------------------------

plt.subplot(2,2,3)

plt.hist(est_f3_rect_cal, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_f3_ham_cal, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_f3_black_cal, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_f3_flat_cal, bins=bins_hist, alpha=transp, label='Flat-top')

# frecuencia verdadera media expresada en bins
plt.axvline(np.mean(f_real), color='black', linestyle='--',
            linewidth=2, label='Frecuencia verdadera media')

plt.title('Frecuencia calibrada - SNR = 3 dB')
plt.xlabel(r'Frecuencia estimada [múltiplos de $\Delta f$]')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA - SNR = 10 dB
# -------------------------

plt.subplot(2,2,4)

plt.hist(est_f10_rect_cal, bins=bins_hist, alpha=transp, label='Rectangular')
plt.hist(est_f10_ham_cal, bins=bins_hist, alpha=transp, label='Hamming')
plt.hist(est_f10_black_cal, bins=bins_hist, alpha=transp, label='Blackman-Harris')
plt.hist(est_f10_flat_cal, bins=bins_hist, alpha=transp, label='Flat-top')

# frecuencia verdadera media expresada en bins
plt.axvline(np.mean(f_real), color='black', linestyle='--',
            linewidth=2, label='Frecuencia verdadera media')

plt.title('Frecuencia calibrada - SNR = 10 dB')
plt.xlabel(r'Frecuencia estimada [múltiplos de $\Delta f$]')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


plt.tight_layout()
plt.show()

#%% BONUS - EFECTO DEL ZERO PADDING SOBRE EL ESTIMADOR DE FRECUENCIA

# analizamos solamente SNR = 10 dB
# usamos Flat-top para comparar antes y despues del zero-padding

N_zp = 10 * N

# -------------------------
# FFT SIN ZERO PADDING
# -------------------------

# ya la tenemos calculada:
# mod10_flat --> tamaño [N/2 , R]

est_f10_flat_sin_zp = np.argmax(mod10_flat, axis=0)


# -------------------------
# FFT CON ZERO PADDING
# -------------------------

# n=N_zp agrega ceros implicitamente hasta completar 10*N muestras
fft10_flat_zp = np.fft.fft(x10_flat, n=N_zp, axis=0)

# mitad positiva del espectro
mod10_flat_zp = np.abs(fft10_flat_zp[:N_zp//2, :])

# buscamos el maximo para cada realizacion
k_zp = np.argmax(mod10_flat_zp, axis=0)

# expresamos nuevamente el resultado en multiplos de Delta f ORIGINAL
# como tenemos 10 veces mas puntos, cada nuevo paso equivale a 0.1 Delta f
est_f10_flat_zp = k_zp * N / N_zp

# -------------------------
# EJES DE FRECUENCIA
# -------------------------
# sin zero padding: pasos de 1 Delta f
k_sin_zp = np.arange(N//2)

# con zero padding: pasos de 0.1 Delta f
k_con_zp = np.arange(N_zp//2) * N / N_zp


# -------------------------
# GRAFICOS
# -------------------------

i = 0   # misma realizacion para comparar


# 2) HISTOGRAMA DEL ESTIMADOR DE FRECUENCIA
plt.figure()

plt.hist(est_f10_flat_sin_zp,
         bins=15, alpha=0.5,
         label='Sin zero-padding')

plt.hist(est_f10_flat_zp,
         bins=15, alpha=0.5,
         label='Con zero-padding (10N)')

plt.axvline(np.mean(f_real), color='black',
            linestyle='--', linewidth=2,
            label='Frecuencia verdadera media')

plt.title('Estimación de frecuencia')
plt.xlabel(r'Frecuencia estimada [múltiplos de $\Delta f$]')
plt.ylabel('Cantidad de realizaciones')

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#%% ================== ESTIMADORES ALTERNATIVOS ==================

# Analizamos solamente SNR = 10 dB


# =========================================================
# ESTIMADOR ALTERNATIVO DE AMPLITUD: RMS
# =========================================================

# Para una senoidal pura:
# RMS = A/sqrt(2)
# entonces A = sqrt(2)*RMS

# Cada columna de x_10 corresponde a una realizacion
# Calculamos un estimador de amplitud para cada una

est_a_rms = np.sqrt(2) * np.sqrt(np.mean(x_10**2, axis=0))


# =========================================================
# ESTIMADOR ALTERNATIVO DE FRECUENCIA:
# INTERPOLACION CUADRATICA DEL PICO (QIP)
# =========================================================

def estimador_qip(X):

    # Nos quedamos con la mitad positiva del espectro
    X_pos = np.abs(X[:len(X)//2])

    # Buscamos el bin donde se encuentra el maximo
    k = np.argmax(X_pos)

    # Si el maximo esta en un extremo no podemos interpolar
    if k == 0 or k == len(X_pos)-1:
        return k

    # Tomamos los tres puntos alrededor del maximo
    alpha = X_pos[k-1]
    beta  = X_pos[k]
    gamma = X_pos[k+1]

    # Interpolacion parabolica
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)

    # Posicion interpolada del maximo
    k_interp = k + p

    return k_interp


# =========================================================
# QIP PARA LAS 200 REALIZACIONES
# =========================================================

est_f_qip = np.zeros(R)

for i in range(R):

    # FFT sin zero-padding
    # De esta manera analizamos solamente el efecto
    # del nuevo estimador
    X = np.fft.fft(x_10[:, i], n=N)

    est_f_qip[i] = estimador_qip(X)


#%% ESTADISTICAS - ESTIMADORES ALTERNATIVOS


# -------------------------
# AMPLITUD RMS
# -------------------------

media_a_rms = np.mean(est_a_rms)

sesgo_a_rms = media_a_rms - a0

var_a_rms = np.var(est_a_rms)


# -------------------------
# FRECUENCIA QIP
# -------------------------

media_f_qip = np.mean(est_f_qip)

sesgo_f_qip = media_f_qip - np.mean(f_real)

var_f_qip = np.var(est_f_qip)


#%% EJEMPLO DE UNA REALIZACION

i = 0

print("\n" + "="*60)
print("ESTIMADORES ALTERNATIVOS - REALIZACION 0")
print("="*60)


print("\nAMPLITUD")
print("-"*60)

print(f"Valor verdadero:          {a0:.4f}")
print(f"Estimador FFT original:   {est_a10_rect[i]:.4f}")
print(f"Estimador RMS:            {est_a_rms[i]:.4f}")


print("\nFRECUENCIA")
print("-"*60)

print(f"Frecuencia verdadera:     {f_real[i]:.4f}")
print(f"Estimador argmax:         {est_f10_rect[i]:.4f}")
print(f"Estimador QIP:            {est_f_qip[i]:.4f}")

print(f"\nError argmax:             {abs(est_f10_rect[i] - f_real[i]):.4f}")
print(f"Error QIP:                {abs(est_f_qip[i] - f_real[i]):.4f}")


#%% COMPARACION MEDIANTE HISTOGRAMAS

plt.figure(figsize=(13,5))


# -------------------------
# AMPLITUD
# -------------------------

plt.subplot(1,2,1)

plt.hist(est_a10_rect,
         bins=15,
         alpha=0.5,
         label='Original: FFT en k0')

plt.hist(est_a_rms,
         bins=15,
         alpha=0.5,
         label='Alternativo: RMS')

plt.axvline(a0,
            color='black',
            linestyle='--',
            linewidth=2,
            label=fr'$a_0={a0:.3f}$')

plt.axvline(np.mean(est_a10_rect),
            color='C0',
            linestyle=':',
            linewidth=2,
            label='Media original')

plt.axvline(media_a_rms,
            color='C1',
            linestyle=':',
            linewidth=2,
            label='Media RMS')

plt.title('Estimadores de amplitud - SNR = 10 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA
# -------------------------

plt.subplot(1,2,2)

plt.hist(est_f10_rect,
         bins=15,
         alpha=0.5,
         label='Original: argmax')

plt.hist(est_f_qip,
         bins=15,
         alpha=0.5,
         label='Alternativo: QIP')

plt.axvline(np.mean(f_real),
            color='black',
            linestyle='--',
            linewidth=2,
            label='Frecuencia verdadera media')

plt.axvline(np.mean(est_f10_rect),
            color='C0',
            linestyle=':',
            linewidth=2,
            label='Media argmax')

plt.axvline(media_f_qip,
            color='C1',
            linestyle=':',
            linewidth=2,
            label='Media QIP')

plt.title('Estimadores de frecuencia - SNR = 10 dB')
plt.xlabel(r'Frecuencia estimada [múltiplos de $\Delta f$]')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


plt.tight_layout()
plt.show()


#%% RESULTADOS ESTIMADORES ALTERNATIVOS

print("\n" + "="*70)
print("COMPARACION DE ESTIMADORES - SNR = 10 dB")
print("="*70)


# -------------------------
# AMPLITUD
# -------------------------

print("\nESTIMACION DE AMPLITUD")
print("-"*70)

print(f"{'Estimador':<25} {'Media':>12} {'Sesgo':>12} {'Varianza':>12}")

print("-"*70)

print(f"{'Original FFT':<25} "
      f"{media_a10_rect:>12.6f} "
      f"{sesgo_a10_rect:>12.6f} "
      f"{var_a10_rect:>12.6f}")

print(f"{'RMS':<25} "
      f"{media_a_rms:>12.6f} "
      f"{sesgo_a_rms:>12.6f} "
      f"{var_a_rms:>12.6f}")


# -------------------------
# FRECUENCIA
# -------------------------

print("\nESTIMACION DE FRECUENCIA")
print("-"*70)

print(f"{'Estimador':<25} {'Media':>12} {'Sesgo':>12} {'Varianza':>12}")

print("-"*70)

print(f"{'Original argmax':<25} "
      f"{media_f10_rect:>12.6f} "
      f"{sesgo_f10_rect:>12.6f} "
      f"{var_f10_rect:>12.6f}")

print(f"{'QIP':<25} "
      f"{media_f_qip:>12.6f} "
      f"{sesgo_f_qip:>12.6f} "
      f"{var_f_qip:>12.6f}")
#%% ESPECTROS DE LAS 200 REALIZACIONES - SNR = 3 dB

k = np.arange(N//2)

plt.figure(figsize=(14,10))

# -------------------------
# RECTANGULAR
# -------------------------

plt.subplot(2,2,1)

for i in range(R):
    X_db = 20*np.log10(mod3_rect[:,i] / np.max(mod3_rect[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Rectangular')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()


# -------------------------
# FLAT-TOP
# -------------------------

plt.subplot(2,2,2)

for i in range(R):
    X_db = 20*np.log10(mod3_flat[:,i] / np.max(mod3_flat[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Flat-top')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()


# -------------------------
# BLACKMAN-HARRIS
# -------------------------

plt.subplot(2,2,3)

for i in range(R):
    X_db = 20*np.log10(mod3_black[:,i] / np.max(mod3_black[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Blackman-Harris')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()


# -------------------------
# HAMMING
# -------------------------

plt.subplot(2,2,4)

for i in range(R):
    X_db = 20*np.log10(mod3_ham[:,i] / np.max(mod3_ham[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Hamming')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()

plt.suptitle('Espectros de las 200 realizaciones - SNR = 3 dB')

plt.tight_layout()
plt.show()

#%% ESPECTROS DE LAS 200 REALIZACIONES - SNR = 10 dB

k = np.arange(N//2)

plt.figure(figsize=(14,10))

# -------------------------
# RECTANGULAR
# -------------------------

plt.subplot(2,2,1)

for i in range(R):
    X_db = 20*np.log10(mod10_rect[:,i] / np.max(mod10_rect[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Rectangular')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()


# -------------------------
# FLAT-TOP
# -------------------------

plt.subplot(2,2,2)

for i in range(R):
    X_db = 20*np.log10(mod10_flat[:,i] / np.max(mod10_flat[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Flat-top')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()


# -------------------------
# BLACKMAN-HARRIS
# -------------------------

plt.subplot(2,2,3)

for i in range(R):
    X_db = 20*np.log10(mod10_black[:,i] / np.max(mod10_black[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Blackman-Harris')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()


# -------------------------
# HAMMING
# -------------------------

plt.subplot(2,2,4)

for i in range(R):
    X_db = 20*np.log10(mod10_ham[:,i] / np.max(mod10_ham[:,i]) + 1e-12)
    plt.plot(k, X_db, alpha=0.25)

plt.axvline(k0, color='black', linestyle='--', linewidth=1.5)

plt.title('Hamming')
plt.xlabel('Bin k')
plt.ylabel('Magnitud [dB]')
plt.xlim(k0-20, k0+20)
plt.ylim(-100,5)
plt.grid()

plt.suptitle('Espectros de las 200 realizaciones - SNR = 10 dB')

plt.tight_layout()
plt.show()


#%% HISTOGRAMAS - ESTIMADORES ORIGINALES

bins_hist = 15
transp = 0.5

# Colores para mantener coherencia entre histograma y media
c_rect = 'C0'
c_ham = 'C1'
c_black = 'C2'
c_flat = 'C3'

plt.figure(figsize=(14,10))


# -------------------------
# AMPLITUD - SNR = 3 dB
# -------------------------

plt.subplot(2,2,1)

plt.hist(est_a3_rect, bins=bins_hist, alpha=transp,
         color=c_rect, label='Rectangular')

plt.hist(est_a3_ham, bins=bins_hist, alpha=transp,
         color=c_ham, label='Hamming')

plt.hist(est_a3_black, bins=bins_hist, alpha=transp,
         color=c_black, label='Blackman-Harris')

plt.hist(est_a3_flat, bins=bins_hist, alpha=transp,
         color=c_flat, label='Flat-top')


# Medias
plt.axvline(np.mean(est_a3_rect), color=c_rect,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_a3_ham), color=c_ham,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_a3_black), color=c_black,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_a3_flat), color=c_flat,
            linestyle=':', linewidth=2)


# Verdadero valor de amplitud
plt.axvline(a0, color='black', linestyle='--',
            linewidth=2, label=fr'$a_0={a0:.3f}$')

plt.title('Estimación de amplitud - SNR = 3 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# AMPLITUD - SNR = 10 dB
# -------------------------

plt.subplot(2,2,2)

plt.hist(est_a10_rect, bins=bins_hist, alpha=transp,
         color=c_rect, label='Rectangular')

plt.hist(est_a10_ham, bins=bins_hist, alpha=transp,
         color=c_ham, label='Hamming')

plt.hist(est_a10_black, bins=bins_hist, alpha=transp,
         color=c_black, label='Blackman-Harris')

plt.hist(est_a10_flat, bins=bins_hist, alpha=transp,
         color=c_flat, label='Flat-top')


# Medias
plt.axvline(np.mean(est_a10_rect), color=c_rect,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_a10_ham), color=c_ham,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_a10_black), color=c_black,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_a10_flat), color=c_flat,
            linestyle=':', linewidth=2)


plt.axvline(a0, color='black', linestyle='--',
            linewidth=2, label=fr'$a_0={a0:.3f}$')

plt.title('Estimación de amplitud - SNR = 10 dB')
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA - SNR = 3 dB
# -------------------------

plt.subplot(2,2,3)

plt.hist(est_f3_rect, bins=bins_hist, alpha=transp,
         color=c_rect, label='Rectangular')

plt.hist(est_f3_ham, bins=bins_hist, alpha=transp,
         color=c_ham, label='Hamming')

plt.hist(est_f3_black, bins=bins_hist, alpha=transp,
         color=c_black, label='Blackman-Harris')

plt.hist(est_f3_flat, bins=bins_hist, alpha=transp,
         color=c_flat, label='Flat-top')


# Medias
plt.axvline(np.mean(est_f3_rect), color=c_rect,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_f3_ham), color=c_ham,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_f3_black), color=c_black,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_f3_flat), color=c_flat,
            linestyle=':', linewidth=2)


plt.axvline(k0, color='black', linestyle='--',
            linewidth=2, label=r'$k_0=N/4$')

plt.title('Estimación de frecuencia - SNR = 3 dB')
plt.xlabel('Bin estimado')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()


# -------------------------
# FRECUENCIA - SNR = 10 dB
# -------------------------

plt.subplot(2,2,4)

plt.hist(est_f10_rect, bins=bins_hist, alpha=transp,
         color=c_rect, label='Rectangular')

plt.hist(est_f10_ham, bins=bins_hist, alpha=transp,
         color=c_ham, label='Hamming')

plt.hist(est_f10_black, bins=bins_hist, alpha=transp,
         color=c_black, label='Blackman-Harris')

plt.hist(est_f10_flat, bins=bins_hist, alpha=transp,
         color=c_flat, label='Flat-top')


# Medias
plt.axvline(np.mean(est_f10_rect), color=c_rect,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_f10_ham), color=c_ham,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_f10_black), color=c_black,
            linestyle=':', linewidth=2)

plt.axvline(np.mean(est_f10_flat), color=c_flat,
            linestyle=':', linewidth=2)


plt.axvline(k0, color='black', linestyle='--',
            linewidth=2, label=r'$k_0=N/4$')

plt.title('Estimación de frecuencia - SNR = 10 dB')
plt.xlabel('Bin estimado')
plt.ylabel('Cantidad de realizaciones')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()