# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:02:13 2026

@author: hoffe
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio


plt.close('all')

#%%FUNCIONES 

def ancho_banda_ecg(f, Pxx, porcentaje=0.99):
    """Calcula la frecuencia superior que contiene el porcentaje
    Para ECG consideramos:f_inf = 0 Hz
        BW = f_sup"""
    df = f[1] - f[0]
    # Potencia acumulada
    potencia_acum = np.cumsum(Pxx) * df
    # Normalización
    potencia_acum = potencia_acum / potencia_acum[-1]
    # Primer punto donde se alcanza el porcentaje deseado
    j = np.where(potencia_acum >= porcentaje)[0][0]
    f_inf = 0
    f_sup = f[j]
    BW = f_sup - f_inf

    return BW, f_inf, f_sup, 0, j

def blackman_tukey(x, fs, M=None, nfft=None):
    x = np.asarray(x).ravel()
    N = len(x)
    if M is None:
        M = N // 5

    if nfft is None:
        nfft = N

    # Quitamos la media, de forma coherente con
    # periodogram() y welch()
    x = x - np.mean(x)

    # Autocorrelación de TODA la señal
    r_full = sig.correlate(x,x,mode='full',method='fft') / N
    # Lag cero
    centro = N - 1
    # Nos quedamos con los lags entre -M y +M
    r = r_full[centro-M : centro+M+1]
    # Ventaneamos la autocorrelación
    ventana = sig.windows.blackman(len(r))
    r_vent = r * ventana
    # Acomodamos los lags para realizar la FFT
    r_fft = np.zeros(nfft)
    r_fft[:M+1] = r_vent[M:]  # lag 0 ... +M
    r_fft[-M:] = r_vent[:M]    # lag -M ... -1
    Pxx = np.abs(np.fft.rfft(r_fft)) / fs# Transformada de Fourier de la autocorrelación
    # PSD unilateral
    if nfft % 2 == 0:
        Pxx[1:-1] *= 2
    else:
        Pxx[1:] *= 2
    f = np.fft.rfftfreq(nfft, d=1/fs)
    return f, Pxx

#FUNCION ANCHO DE BANDA PARA AUDIO

def ancho_banda_audio(f, Pxx, porcentaje=0.99):

    df = f[1] - f[0]
    potencia = Pxx * df

    potencia_total = np.sum(potencia)
    objetivo = porcentaje * potencia_total

    acum = np.concatenate(([0], np.cumsum(potencia)))

    mejor_BW = np.inf
    mejor_i = 0
    mejor_j = len(f)-1

    for i in range(len(f)):

        objetivo_j = acum[i] + objetivo
        j = np.searchsorted(acum, objetivo_j)

        if j > len(f):
            break

        j = j - 1

        BW = f[j] - f[i]

        if BW < mejor_BW:
            mejor_BW = BW
            mejor_i = i
            mejor_j = j

    f_inf = f[mejor_i]
    f_sup = f[mejor_j]
    BW = f_sup - f_inf

    return BW, f_inf, f_sup, mejor_i, mejor_j
#%% ECG

# Lectura de ECG #
fs_ecg = 1000  # Hz

##################
# ECG sin ruido  #
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

N = len(ecg_one_lead)
t_ecg = np.arange(N) / fs_ecg
print('Cantidad de muestras:', N)
print('Duración:', N/fs_ecg, 's')


#%% PSD DEL ECG - MÉTODO DE WELCH

# Se comienza con pocos promedios (mayor varianza)
# y se aumenta progresivamente la cantidad de promedios.

valores_nperseg = [8192, 1905, 1792, 1644, 1446]
#valores_nperseg = [10000, 8192, 4096, 2048, 3000]
plt.figure(figsize=(12,7))

for nperseg in valores_nperseg:

    noverlap = nperseg // 2

    # Cantidad REAL de segmentos promediados
    K = 1 + (N - nperseg) // (nperseg - noverlap)

    # Separación entre bins
    df = fs_ecg / nperseg

    # Welch
    f, psd = sig.welch(ecg_one_lead,fs=fs_ecg,window='hann',nperseg=nperseg,noverlap=noverlap)

    plt.plot(f,psd,label=f'{K} promedios  (nperseg={nperseg})')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('ECG - Comparación de cantidad de promedios en Welch')
plt.xlim(0,60)
plt.grid()
plt.legend()
plt.show()


#%%PSD ECG - Welch elegido: 30 promedios   #

nperseg_ecg = 1905
noverlap_ecg = nperseg_ecg // 2

f_welch_ecg, Pxx_welch_ecg = sig.welch(ecg_one_lead,fs=fs_ecg,window='hann',nperseg=nperseg_ecg,noverlap=noverlap_ecg,
    scaling='density')
K = 1 + (N - nperseg_ecg) // (nperseg_ecg - noverlap_ecg)
print('Welch - cantidad de promedios:', K)
BW_welch_ecg, f_welch_inf, f_welch_sup, \
i_welch, j_welch = ancho_banda_ecg(f_welch_ecg,Pxx_welch_ecg)

#%% PSD ECG - PERIODOGRAMA MODIFICADO


f_per_ecg, Pxx_per_ecg = sig.periodogram(ecg_one_lead,fs=fs_ecg,scaling='density')

BW_per_ecg, f_per_inf, f_per_sup, \
i_per, j_per = ancho_banda_ecg(f_per_ecg,Pxx_per_ecg)

#%% PSD ECG - BLACKMAN TUKEY

M_ecg = 2000
f_bt_ecg, Pxx_bt_ecg = blackman_tukey(ecg_one_lead,fs=fs_ecg, M=M_ecg,nfft=N)

BW_bt_ecg, f_bt_inf, f_bt_sup, \
i_bt, j_bt = ancho_banda_ecg(f_bt_ecg,Pxx_bt_ecg)

#%% FIGURA COMPARATIVA ECG

fig, axs = plt.subplots(4, 1, figsize=(12, 14))

# Señal temporal
axs[0].plot(t_ecg, ecg_one_lead)
axs[0].set_title('ECG sin ruido - Dominio temporal')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].grid()

# Welch
axs[1].plot(f_welch_ecg, Pxx_welch_ecg, label='PSD')
axs[1].axvline(f_welch_inf, ls='--', label=f'f_inf = {f_welch_inf:.2f} Hz')
axs[1].axvline(f_welch_sup, ls='--', label=f'f_sup = {f_welch_sup:.2f} Hz')
axs[1].fill_between(f_welch_ecg[i_welch:j_welch+1], Pxx_welch_ecg[i_welch:j_welch+1], alpha=0.3, label='99% potencia')
axs[1].set_title(f'PSD ECG - Welch - 30 promedios - BW = {BW_welch_ecg:.2f} Hz')
axs[1].set_xlim(-2, 60)
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('PSD')
axs[1].grid()
axs[1].legend()

# Periodograma modificado
axs[2].plot(f_per_ecg, Pxx_per_ecg, label='PSD')
axs[2].axvline(f_per_inf, ls='--', label=f'f_inf = {f_per_inf:.2f} Hz')
axs[2].axvline(f_per_sup, ls='--', label=f'f_sup = {f_per_sup:.2f} Hz')
axs[2].fill_between(f_per_ecg[i_per:j_per+1], Pxx_per_ecg[i_per:j_per+1], alpha=0.3, label='99% potencia')
axs[2].set_title(f'PSD ECG - Periodograma modificado - BW = {BW_per_ecg:.2f} Hz')
axs[2].set_xlim(-2, 60)
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('PSD')
axs[2].grid()
axs[2].legend()

# Blackman-Tukey
axs[3].plot(f_bt_ecg, Pxx_bt_ecg, label='PSD')
axs[3].axvline(f_bt_inf, ls='--', label=f'f_inf = {f_bt_inf:.2f} Hz')
axs[3].axvline(f_bt_sup, ls='--', label=f'f_sup = {f_bt_sup:.2f} Hz')
axs[3].fill_between(f_bt_ecg[i_bt:j_bt+1], Pxx_bt_ecg[i_bt:j_bt+1], alpha=0.3, label='99% potencia')
axs[3].set_title(f'PSD ECG - Blackman-Tukey - BW = {BW_bt_ecg:.2f} Hz')
axs[3].set_xlim(-2, 60)
axs[3].set_xlabel('Frecuencia [Hz]')
axs[3].set_ylabel('PSD')
axs[3].grid()
axs[3].legend()

plt.tight_layout()
plt.show()



#%% PPG 2DA SENAL
####################################
# Lectura de pletismografía (PPG)  #
####################################
fs_ppg = 400  # Hz
ppg = np.load('ppg_sin_ruido.npy')

N_ppg = len(ppg)
t_ppg = np.arange(N_ppg) / fs_ppg

print('\nPPG')
print('Cantidad de muestras:', N_ppg)
print('Duración:', N_ppg/fs_ppg, 's')


def nperseg_para_promedios(N, K_deseado):
    posibles = []
    for nperseg in range(2, N+1):
        noverlap = nperseg // 2
        K = 1 + (N - nperseg) // (nperseg - noverlap)
        if K == K_deseado:
            posibles.append(nperseg)
    return max(posibles) if posibles else None


#%% PPG - COMPARACIÓN DE PROMEDIOS EN WELCH

promedios_ppg = [5, 10, 20, 30, 40]

plt.figure(figsize=(12,7))

for K_deseado in promedios_ppg:
    nperseg = nperseg_para_promedios(N_ppg, K_deseado)
    noverlap = nperseg // 2

    f, Pxx = sig.welch(ppg, fs=fs_ppg, window='hann',
                       nperseg=nperseg, noverlap=noverlap)

    K_real = 1 + (N_ppg-nperseg)//(nperseg-noverlap)

    plt.plot(f, Pxx, label=f'{K_real} promedios')

plt.xlim(0,10)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('PPG - Comparación de cantidad de promedios en Welch')
plt.grid()
plt.legend()
plt.show()

#%% PPG - COMPARACIÓN DE M EN BLACKMAN-TUKEY

valores_M_ppg = [N_ppg//3, N_ppg//5, N_ppg//10, N_ppg//20, N_ppg//40]

plt.figure(figsize=(12,7))

for M in valores_M_ppg:
    f_bt, Pxx_bt = blackman_tukey(ppg, fs_ppg, M=M, nfft=N_ppg)
    plt.plot(f_bt, Pxx_bt, label=f'M = {M}')

plt.xlim(0,10)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('PPG - Blackman-Tukey: comparación de M')
plt.grid()
plt.legend()
plt.show()

#%% PPG - WELCH DEFINITIVO: 32 PROMEDIOS

nperseg_ppg = nperseg_para_promedios(N_ppg, 32)
noverlap_ppg = nperseg_ppg // 2

f_welch_ppg, Pxx_welch_ppg = sig.welch(
    ppg,fs=fs_ppg,window='hann',nperseg=nperseg_ppg,noverlap=noverlap_ppg,scaling='density')

K_ppg = 1 + (N_ppg-nperseg_ppg)//(nperseg_ppg-noverlap_ppg)



BW_welch_ppg, f_welch_ppg_inf, f_welch_ppg_sup, \
i_welch_ppg, j_welch_ppg = ancho_banda_ecg(
    f_welch_ppg, Pxx_welch_ppg)

#%% PPG - BLACKMAN TUKEY DEFINITIVO

M_ppg = 1122
f_bt_ppg, Pxx_bt_ppg = blackman_tukey(ppg,fs=fs_ppg,M=M_ppg,nfft=N_ppg)

BW_bt_ppg, f_bt_ppg_inf, f_bt_ppg_sup, \
i_bt_ppg, j_bt_ppg = ancho_banda_ecg(f_bt_ppg, Pxx_bt_ppg)


#%% FIGURA FINAL PPG

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Señal temporal
axs[0].plot(t_ppg, ppg)
axs[0].set_title('PPG sin ruido - Dominio temporal')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].grid()

# Welch
axs[1].plot(f_welch_ppg, Pxx_welch_ppg, label='PSD')
axs[1].axvline(f_welch_ppg_inf, ls='--',
               label=f'f_inf = {f_welch_ppg_inf:.2f} Hz')
axs[1].axvline(f_welch_ppg_sup, ls='--',
               label=f'f_sup = {f_welch_ppg_sup:.2f} Hz')
axs[1].fill_between(
    f_welch_ppg[i_welch_ppg:j_welch_ppg+1],
    Pxx_welch_ppg[i_welch_ppg:j_welch_ppg+1],
    alpha=0.3, label='99% potencia'
)
axs[1].set_title(
    f'PSD PPG - Welch - {K_ppg} promedios - BW = {BW_welch_ppg:.2f} Hz'
)
axs[1].set_xlim(-0.5, 10)
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('PSD')
axs[1].grid()
axs[1].legend()

# Blackman-Tukey
axs[2].plot(f_bt_ppg, Pxx_bt_ppg, label='PSD')
axs[2].axvline(f_bt_ppg_inf, ls='--',
               label=f'f_inf = {f_bt_ppg_inf:.2f} Hz')
axs[2].axvline(f_bt_ppg_sup, ls='--',
               label=f'f_sup = {f_bt_ppg_sup:.2f} Hz')
axs[2].fill_between(
    f_bt_ppg[i_bt_ppg:j_bt_ppg+1],
    Pxx_bt_ppg[i_bt_ppg:j_bt_ppg+1],
    alpha=0.3, label='99% potencia'
)
axs[2].set_title(
    f'PSD PPG - Blackman-Tukey - M = {M_ppg} - BW = {BW_bt_ppg:.2f} Hz'
)
axs[2].set_xlim(-0.5, 10)
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('PSD')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()

#%% AUDIO 1 - LA CUCARACHA

fs_audio1, audio1 = sio.wavfile.read('la cucaracha.wav')

# Si el audio tiene dos canales, lo pasamos a mono
if audio1.ndim > 1:
    audio1 = np.mean(audio1, axis=1)

audio1 = audio1.astype(float)

N_audio1 = len(audio1)
t_audio1 = np.arange(N_audio1) / fs_audio1

print('\nLA CUCARACHA')
print('Frecuencia de muestreo:', fs_audio1, 'Hz')
print('Cantidad de muestras:', N_audio1)
print('Duración:', N_audio1/fs_audio1, 's')


#%% AUDIO 1 - COMPARACION DE PROMEDIOS EN WELCH

promedios_audio1 = [ 20, 30, 40, 50]

plt.figure(figsize=(12,7))

for K_deseado in promedios_audio1:
    nperseg = nperseg_para_promedios(N_audio1, K_deseado)
    noverlap = nperseg // 2

    f, Pxx = sig.welch(
        audio1,
        fs=fs_audio1,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    K_real = 1 + (N_audio1-nperseg)//(nperseg-noverlap)

    plt.plot(f, Pxx, label=f'{K_real} promedios')


plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('La cucaracha - Comparación de promedios en Welch')
plt.grid()
plt.legend()
plt.xlim(0,2500)
plt.show()

#%% AUDIO 1 - COMPARACION DE M EN BLACKMAN-TUKEY

valores_M_audio1 = [
    N_audio1//3,
    N_audio1//5,
    N_audio1//10,
    N_audio1//20,
    N_audio1//40]

plt.figure(figsize=(12,7))

for M in valores_M_audio1:
    f_bt, Pxx_bt = blackman_tukey(
        audio1,
        fs_audio1,
        M=M,
        nfft=N_audio1
    )

    plt.plot(f_bt, Pxx_bt, label=f'M = {M}')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('La cucaracha - Comparación de M en Blackman-Tukey')
plt.grid()
plt.legend()
plt.xlim(0,2500)
plt.show()

#%% AUDIO 1 - WELCH DEFINITIVO: 50 PROMEDIOS

nperseg_audio1 = nperseg_para_promedios(N_audio1, 50)
noverlap_audio1 = nperseg_audio1 // 2

f_welch_audio1, Pxx_welch_audio1 = sig.welch(
    audio1,
    fs=fs_audio1,
    window='hann',
    nperseg=nperseg_audio1,
    noverlap=noverlap_audio1,
    scaling='density'
)

K_audio1 = 1 + (N_audio1-nperseg_audio1)//(nperseg_audio1-noverlap_audio1)

BW_welch_audio1, f_welch_audio1_inf, f_welch_audio1_sup, \
i_welch_audio1, j_welch_audio1 = ancho_banda_audio(
    f_welch_audio1, Pxx_welch_audio1
)

print('Welch - cantidad de promedios:', K_audio1)
print('Welch - nperseg:', nperseg_audio1)

#%% AUDIO 1 - BLACKMAN TUKEY DEFINITIVO

M_audio1 = 3600

f_bt_audio1, Pxx_bt_audio1 = blackman_tukey(
    audio1,
    fs=fs_audio1,
    M=M_audio1,
    nfft=N_audio1
)

BW_bt_audio1, f_bt_audio1_inf, f_bt_audio1_sup, \
i_bt_audio1, j_bt_audio1 = ancho_banda_audio(
    f_bt_audio1, Pxx_bt_audio1
)


#%% FIGURA FINAL - LA CUCARACHA

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Señal temporal
axs[0].plot(t_audio1, audio1)
axs[0].set_title('La cucaracha - Dominio temporal')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].grid()

# Welch
axs[1].plot(f_welch_audio1, Pxx_welch_audio1, label='PSD')
axs[1].axvline(f_welch_audio1_inf, ls='--',
               label=f'f_inf = {f_welch_audio1_inf:.2f} Hz')
axs[1].axvline(f_welch_audio1_sup, ls='--',
               label=f'f_sup = {f_welch_audio1_sup:.2f} Hz')
axs[1].fill_between(
    f_welch_audio1[i_welch_audio1:j_welch_audio1+1],
    Pxx_welch_audio1[i_welch_audio1:j_welch_audio1+1],
    alpha=0.3, label='99% potencia'
)
axs[1].set_title(
    f'PSD - Welch - {K_audio1} promedios - BW = {BW_welch_audio1:.2f} Hz'
)
axs[1].set_xlim(0, f_welch_audio1_sup*1.2)
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('PSD')
axs[1].grid()
axs[1].legend()

# Blackman-Tukey
axs[2].plot(f_bt_audio1, Pxx_bt_audio1, label='PSD')
axs[2].axvline(f_bt_audio1_inf, ls='--',
               label=f'f_inf = {f_bt_audio1_inf:.2f} Hz')
axs[2].axvline(f_bt_audio1_sup, ls='--',
               label=f'f_sup = {f_bt_audio1_sup:.2f} Hz')
axs[2].fill_between(
    f_bt_audio1[i_bt_audio1:j_bt_audio1+1],
    Pxx_bt_audio1[i_bt_audio1:j_bt_audio1+1],
    alpha=0.3, label='99% potencia'
)
axs[2].set_title(
    f'PSD - Blackman-Tukey - M = {M_audio1} - BW = {BW_bt_audio1:.2f} Hz'
)
axs[2].set_xlim(0, f_bt_audio1_sup*1.2)
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('PSD')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()


#%% AUDIO 2 - PRUEBA PSD

fs_audio2, audio2 = sio.wavfile.read('prueba psd.wav')

# Si tiene dos canales, lo pasamos a mono
if audio2.ndim > 1:
    audio2 = np.mean(audio2, axis=1)

audio2 = audio2.astype(float)

N_audio2 = len(audio2)
t_audio2 = np.arange(N_audio2) / fs_audio2

print('\nPRUEBA PSD')
print('Frecuencia de muestreo:', fs_audio2, 'Hz')
print('Cantidad de muestras:', N_audio2)
print('Duración:', N_audio2/fs_audio2, 's')

#%% AUDIO 2 - COMPARACION DE PROMEDIOS EN WELCH

promedios_audio2 = [10, 20, 30, 40, 50, 60]

plt.figure(figsize=(12,7))

for K_deseado in promedios_audio2:
    nperseg = nperseg_para_promedios(N_audio2, K_deseado)
    noverlap = nperseg // 2

    f, Pxx = sig.welch(
        audio2,
        fs=fs_audio2,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    K_real = 1 + (N_audio2-nperseg)//(nperseg-noverlap)

    plt.plot(f, Pxx, label=f'{K_real} promedios')


plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('Prueba PSD - Comparación de promedios en Welch')
plt.grid()
plt.xlim(0,2000)
plt.legend()
plt.show()

#%% AUDIO 2 - COMPARACION DE M EN BLACKMAN-TUKEY

valores_M_audio2 = [
    N_audio2//3,
    N_audio2//5,
    N_audio2//10,
    N_audio2//20,
    N_audio2//40
]

plt.figure(figsize=(12,7))

for M in valores_M_audio2:
    f_bt, Pxx_bt = blackman_tukey(
        audio2,
        fs_audio2,
        M=M,
        nfft=N_audio2
    )

    plt.plot(f_bt, Pxx_bt, label=f'M = {M}')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('Prueba PSD - Comparación de M en Blackman-Tukey')
plt.xlim(0,2000)
plt.grid()
plt.legend()
plt.show()

#%% AUDIO 2 - WELCH DEFINITIVO: 60 PROMEDIOS

nperseg_audio2 = nperseg_para_promedios(N_audio2, 60)
noverlap_audio2 = nperseg_audio2 // 2

f_welch_audio2, Pxx_welch_audio2 = sig.welch(
    audio2, fs=fs_audio2, window='hann',
    nperseg=nperseg_audio2, noverlap=noverlap_audio2,
    scaling='density'
)

K_audio2 = 1 + (N_audio2-nperseg_audio2)//(nperseg_audio2-noverlap_audio2)

BW_welch_audio2, f_welch_audio2_inf, f_welch_audio2_sup, \
i_welch_audio2, j_welch_audio2 = ancho_banda_audio(
    f_welch_audio2, Pxx_welch_audio2
)

print('Welch - cantidad de promedios:', K_audio2)
print('Welch - nperseg:', nperseg_audio2)

#%% AUDIO 2 - BLACKMAN TUKEY DEFINITIVO

M_audio2 = 3600

f_bt_audio2, Pxx_bt_audio2 = blackman_tukey(
    audio2, fs=fs_audio2, M=M_audio2, nfft=N_audio2
)

BW_bt_audio2, f_bt_audio2_inf, f_bt_audio2_sup, \
i_bt_audio2, j_bt_audio2 = ancho_banda_audio(
    f_bt_audio2, Pxx_bt_audio2
)

#%% FIGURA FINAL - PRUEBA PSD

fig, axs = plt.subplots(3, 1, figsize=(12,10))

axs[0].plot(t_audio2, audio2)
axs[0].set_title('Prueba PSD - Dominio temporal')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].grid()

axs[1].plot(f_welch_audio2, Pxx_welch_audio2, label='PSD')
axs[1].axvline(f_welch_audio2_inf, ls='--', label=f'f_inf = {f_welch_audio2_inf:.2f} Hz')
axs[1].axvline(f_welch_audio2_sup, ls='--', label=f'f_sup = {f_welch_audio2_sup:.2f} Hz')
axs[1].fill_between(f_welch_audio2[i_welch_audio2:j_welch_audio2+1],
                    Pxx_welch_audio2[i_welch_audio2:j_welch_audio2+1],
                    alpha=0.3, label='99% potencia')
axs[1].set_title(f'PSD - Welch - {K_audio2} promedios - BW = {BW_welch_audio2:.2f} Hz')
axs[1].set_xlim(0, f_welch_audio2_sup*1.2)
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('PSD')
axs[1].grid()
axs[1].legend()

axs[2].plot(f_bt_audio2, Pxx_bt_audio2, label='PSD')
axs[2].axvline(f_bt_audio2_inf, ls='--', label=f'f_inf = {f_bt_audio2_inf:.2f} Hz')
axs[2].axvline(f_bt_audio2_sup, ls='--', label=f'f_sup = {f_bt_audio2_sup:.2f} Hz')
axs[2].fill_between(f_bt_audio2[i_bt_audio2:j_bt_audio2+1],
                    Pxx_bt_audio2[i_bt_audio2:j_bt_audio2+1],
                    alpha=0.3, label='99% potencia')
axs[2].set_title(f'PSD - Blackman-Tukey - M = {M_audio2} - BW = {BW_bt_audio2:.2f} Hz')
axs[2].set_xlim(0, f_bt_audio2_sup*1.2)
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('PSD')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()

#%% AUDIO 3 - SILBIDO

fs_audio3, audio3 = sio.wavfile.read('silbido.wav')

# Si tiene dos canales, lo pasamos a mono
if audio3.ndim > 1:
    audio3 = np.mean(audio3, axis=1)

audio3 = audio3.astype(float)

N_audio3 = len(audio3)
t_audio3 = np.arange(N_audio3) / fs_audio3

print('\nSILBIDO')
print('Frecuencia de muestreo:', fs_audio3, 'Hz')
print('Cantidad de muestras:', N_audio3)
print('Duración:', N_audio3/fs_audio3, 's')


#%% AUDIO 3 - COMPARACION DE PROMEDIOS EN WELCH

promedios_audio3 = [20, 30, 40, 50, 60, 70]

plt.figure(figsize=(12,7))

for K_deseado in promedios_audio3:

    nperseg = nperseg_para_promedios(N_audio3, K_deseado)
    noverlap = nperseg // 2

    f, Pxx = sig.welch(
        audio3,
        fs=fs_audio3,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    K_real = 1 + (N_audio3-nperseg)//(nperseg-noverlap)

    plt.plot(f, Pxx, label=f'{K_real} promedios')

    print(f'{K_real} promedios -> nperseg = {nperseg} -> '
          f'Delta f = {fs_audio3/nperseg:.2f} Hz')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('Silbido - Comparación de promedios en Welch')
plt.grid()
plt.legend()
plt.xlim(0,8000)
plt.show()

#%% AUDIO 3 - COMPARACION DE M EN BLACKMAN-TUKEY

valores_M_audio3 = [
    N_audio3//3,
    N_audio3//5,
    N_audio3//10,
    N_audio3//20,
    N_audio3//40
]

plt.figure(figsize=(12,7))

for M in valores_M_audio3:

    f_bt, Pxx_bt = blackman_tukey(
        audio3,
        fs_audio3,
        M=M,
        nfft=N_audio3
    )

    plt.plot(f_bt, Pxx_bt, label=f'M = {M}')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('Silbido - Comparación de M en Blackman-Tukey')
plt.grid()
plt.legend()
plt.xlim(0,8000)
plt.show()

#%% AUDIO 3 - WELCH DEFINITIVO: 70 PROMEDIOS

nperseg_audio3 = nperseg_para_promedios(N_audio3, 70)
noverlap_audio3 = nperseg_audio3 // 2

f_welch_audio3, Pxx_welch_audio3 = sig.welch(
    audio3,
    fs=fs_audio3,
    window='hann',
    nperseg=nperseg_audio3,
    noverlap=noverlap_audio3,
    scaling='density'
)

K_audio3 = 1 + (N_audio3-nperseg_audio3)//(nperseg_audio3-noverlap_audio3)

BW_welch_audio3, f_welch_audio3_inf, f_welch_audio3_sup, \
i_welch_audio3, j_welch_audio3 = ancho_banda_audio(
    f_welch_audio3, Pxx_welch_audio3
)

print('Welch - cantidad de promedios:', K_audio3)
print('Welch - nperseg:', nperseg_audio3)

#%% AUDIO 3 - BLACKMAN TUKEY DEFINITIVO

M_audio3 = 3600

f_bt_audio3, Pxx_bt_audio3 = blackman_tukey(
    audio3,
    fs=fs_audio3,
    M=M_audio3,
    nfft=N_audio3
)

BW_bt_audio3, f_bt_audio3_inf, f_bt_audio3_sup, \
i_bt_audio3, j_bt_audio3 = ancho_banda_audio(
    f_bt_audio3, Pxx_bt_audio3
)

#%% FIGURA FINAL - SILBIDO

fig, axs = plt.subplots(3, 1, figsize=(12,10))

# Señal temporal
axs[0].plot(t_audio3, audio3)
axs[0].set_title('Silbido - Dominio temporal')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].grid()

# Welch
axs[1].plot(f_welch_audio3, Pxx_welch_audio3, label='PSD')
axs[1].axvline(f_welch_audio3_inf, ls='--',
               label=f'f_inf = {f_welch_audio3_inf:.2f} Hz')
axs[1].axvline(f_welch_audio3_sup, ls='--',
               label=f'f_sup = {f_welch_audio3_sup:.2f} Hz')
axs[1].fill_between(
    f_welch_audio3[i_welch_audio3:j_welch_audio3+1],
    Pxx_welch_audio3[i_welch_audio3:j_welch_audio3+1],
    alpha=0.3, label='99% potencia'
)
axs[1].set_title(
    f'PSD - Welch - {K_audio3} promedios - BW = {BW_welch_audio3:.2f} Hz'
)
axs[1].set_xlim(0, f_welch_audio3_sup*1.2)
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('PSD')
axs[1].grid()
axs[1].legend()

# Blackman-Tukey
axs[2].plot(f_bt_audio3, Pxx_bt_audio3, label='PSD')
axs[2].axvline(f_bt_audio3_inf, ls='--',
               label=f'f_inf = {f_bt_audio3_inf:.2f} Hz')
axs[2].axvline(f_bt_audio3_sup, ls='--',
               label=f'f_sup = {f_bt_audio3_sup:.2f} Hz')
axs[2].fill_between(
    f_bt_audio3[i_bt_audio3:j_bt_audio3+1],
    Pxx_bt_audio3[i_bt_audio3:j_bt_audio3+1],
    alpha=0.3, label='99% potencia'
)
axs[2].set_title(
    f'PSD - Blackman-Tukey - M = {M_audio3} - BW = {BW_bt_audio3:.2f} Hz'
)
axs[2].set_xlim(0, f_bt_audio3_sup*1.2)
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('PSD')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()

#%% TABLA FINAL DE RESULTADOS

resultados = [
    ['ECG', 'Welch', f_welch_inf, f_welch_sup, BW_welch_ecg],
    ['ECG', 'Blackman-Tukey', f_bt_inf, f_bt_sup, BW_bt_ecg],

    ['PPG', 'Welch', f_welch_ppg_inf, f_welch_ppg_sup, BW_welch_ppg],
    ['PPG', 'Blackman-Tukey', f_bt_ppg_inf, f_bt_ppg_sup, BW_bt_ppg],

    ['La cucaracha', 'Welch', f_welch_audio1_inf, f_welch_audio1_sup, BW_welch_audio1],
    ['La cucaracha', 'Blackman-Tukey', f_bt_audio1_inf, f_bt_audio1_sup, BW_bt_audio1],

    ['Prueba PSD', 'Welch', f_welch_audio2_inf, f_welch_audio2_sup, BW_welch_audio2],
    ['Prueba PSD', 'Blackman-Tukey', f_bt_audio2_inf, f_bt_audio2_sup, BW_bt_audio2],

    ['Silbido', 'Welch', f_welch_audio3_inf, f_welch_audio3_sup, BW_welch_audio3],
    ['Silbido', 'Blackman-Tukey', f_bt_audio3_inf, f_bt_audio3_sup, BW_bt_audio3]
]

print('\nRESULTADOS FINALES\n')

print(f'{"Señal":<16} {"Método":<18} {"f_inf [Hz]":>12} {"f_sup [Hz]":>12} {"BW [Hz]":>12}')
print('-'*75)

for fila in resultados:
    señal, metodo, f_inf, f_sup, BW = fila

    print(f'{señal:<16} {metodo:<18} '
          f'{f_inf:>12.2f} {f_sup:>12.2f} {BW:>12.2f}')
    
    
#%% BONUS - EMG SANO

fs_emg = 4000  # Hz

datos_emg = np.genfromtxt(
    'emg_healthy.txt',
    delimiter=None,      # detecta espacios o tabulaciones
    encoding='latin1',
    invalid_raise=False
)

# Eliminamos posibles filas no numéricas
if datos_emg.ndim == 2:
    datos_emg = datos_emg[np.all(np.isfinite(datos_emg), axis=1)]

t_emg = datos_emg[:,0]
emg = datos_emg[:,1]

N_emg = len(emg)

#%% BONUS EMG - COMPARACION DE PROMEDIOS EN WELCH

promedios_emg = [ 30, 40, 50, 60, 70]

plt.figure(figsize=(12,7))

for K_deseado in promedios_emg:
    nperseg = nperseg_para_promedios(N_emg, K_deseado)
    noverlap = nperseg // 2

    f, Pxx = sig.welch(
        emg, fs=fs_emg, window='hann',
        nperseg=nperseg, noverlap=noverlap,
        scaling='density'
    )

    K_real = 1 + (N_emg-nperseg)//(nperseg-noverlap)

    plt.plot(f, Pxx, label=f'{K_real} promedios')

    print(f'{K_real} promedios -> nperseg = {nperseg} -> '
          f'Delta f = {fs_emg/nperseg:.2f} Hz')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('EMG sano - Comparación de promedios en Welch')
plt.grid()
plt.legend()
plt.xlim(0,1000)
plt.show()

#%% BONUS EMG - COMPARACION DE M EN BLACKMAN-TUKEY

valores_M_emg = [
    N_emg//10,
    N_emg//20,
    N_emg//40
]

plt.figure(figsize=(12,7))

for M in valores_M_emg:
    f_bt, Pxx_bt = blackman_tukey(
        emg, fs=fs_emg, M=M, nfft=N_emg
    )

    plt.plot(f_bt, Pxx_bt, label=f'M = {M}')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('EMG sano - Comparación de M en Blackman-Tukey')
plt.grid()
plt.legend()
plt.xlim(0,1000)
plt.show()

#%% BONUS EMG - WELCH DEFINITIVO

nperseg_emg = nperseg_para_promedios(N_emg, 70)
noverlap_emg = nperseg_emg // 2

f_welch_emg, Pxx_welch_emg = sig.welch(
    emg, fs=fs_emg, window='hann',
    nperseg=nperseg_emg, noverlap=noverlap_emg,
    scaling='density'
)

K_emg = 1 + (N_emg-nperseg_emg)//(nperseg_emg-noverlap_emg)

BW_welch_emg, f_welch_emg_inf, f_welch_emg_sup, \
i_welch_emg, j_welch_emg = ancho_banda_audio(
    f_welch_emg, Pxx_welch_emg
)

print('Welch - cantidad de promedios:', K_emg)
print('Welch - nperseg:', nperseg_emg)

#%% BONUS EMG - BLACKMAN TUKEY DEFINITIVO

M_emg = 1271

f_bt_emg, Pxx_bt_emg = blackman_tukey(
    emg, fs=fs_emg, M=M_emg, nfft=N_emg
)

BW_bt_emg, f_bt_emg_inf, f_bt_emg_sup, \
i_bt_emg, j_bt_emg = ancho_banda_audio(
    f_bt_emg, Pxx_bt_emg
)

#%% FIGURA FINAL - EMG SANO

fig, axs = plt.subplots(3, 1, figsize=(12,10))

axs[0].plot(t_emg, emg)
axs[0].set_title('EMG sano - Dominio temporal')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].grid()

axs[1].plot(f_welch_emg, Pxx_welch_emg, label='PSD')
axs[1].axvline(f_welch_emg_inf, ls='--',
               label=f'f_inf = {f_welch_emg_inf:.2f} Hz')
axs[1].axvline(f_welch_emg_sup, ls='--',
               label=f'f_sup = {f_welch_emg_sup:.2f} Hz')
axs[1].fill_between(
    f_welch_emg[i_welch_emg:j_welch_emg+1],
    Pxx_welch_emg[i_welch_emg:j_welch_emg+1],
    alpha=0.3, label='99% potencia'
)
axs[1].set_title(
    f'PSD - Welch - {K_emg} promedios - BW = {BW_welch_emg:.2f} Hz'
)
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('PSD')
axs[1].grid()
axs[1].legend()

axs[2].plot(f_bt_emg, Pxx_bt_emg, label='PSD')
axs[2].axvline(f_bt_emg_inf, ls='--',
               label=f'f_inf = {f_bt_emg_inf:.2f} Hz')
axs[2].axvline(f_bt_emg_sup, ls='--',
               label=f'f_sup = {f_bt_emg_sup:.2f} Hz')
axs[2].fill_between(
    f_bt_emg[i_bt_emg:j_bt_emg+1],
    Pxx_bt_emg[i_bt_emg:j_bt_emg+1],
    alpha=0.3, label='99% potencia'
)
axs[2].set_title(
    f'PSD - Blackman-Tukey - M = {M_emg} - BW = {BW_bt_emg:.2f} Hz'
)
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('PSD')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()
#%% RESULTADOS BONUS EMG

print('\nANCHO DE BANDA - EMG SANO')

print(f'Welch:')
print(f'  f_inf = {f_welch_emg_inf:.2f} Hz')
print(f'  f_sup = {f_welch_emg_sup:.2f} Hz')
print(f'  BW    = {BW_welch_emg:.2f} Hz')

print(f'\nBlackman-Tukey:')
print(f'  f_inf = {f_bt_emg_inf:.2f} Hz')
print(f'  f_sup = {f_bt_emg_sup:.2f} Hz')
print(f'  BW    = {BW_bt_emg:.2f} Hz')

#%% POTENCIA ACUMULADA EMG

df = f_welch_emg[1] - f_welch_emg[0]
potencia_acum = np.cumsum(Pxx_welch_emg) * df
potencia_acum /= potencia_acum[-1]

for freq in [100, 250, 500, 750, 1000, 1250, 1500, 1670]:
    i = np.argmin(np.abs(f_welch_emg - freq))
    print(f'Hasta {f_welch_emg[i]:.1f} Hz: '
          f'{potencia_acum[i]*100:.2f} % de la potencia')
