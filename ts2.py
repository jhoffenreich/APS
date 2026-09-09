# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:23:51 2026

@author: hoffe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%% DEFINO PARAMETROS

fs = 1000
N = 1000
B =16
vfs = 2
kn =1

delta_f = fs/N
f0 = delta_f#sera la frecuencia de mi senoidal

#%% FUNCIONES
def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(0, nn/fs, 1/fs)

    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)

    return tt, xx

#%% SENAL SENOIDAL

vmax = np.sqrt(2)#para que tenga potencia unitaria

tt, s = mi_funcion_sen(vmax=vmax,dc=0,ff=f0,ph=0,nn=N,fs=fs)

Ps = np.var(s)

print("Potencia de s =", Ps)

#%% ADC

q = (2*vfs)/(2**B)
Pq = q**2 / 12 #potencia del ruido cuantizado
Pn = kn * Pq # potencia del ruido analogico

sigma_n = np.sqrt(Pn)#desvio estandar, para generar el ruido analogico

n = np.random.normal(0,sigma_n,N)#generamos el ruido analg, con distrib gaussiana, de media 0 y varianza=

print("Potencia medida de n =", np.var(n))
print("Potencia teorica de n =", Pn)

sR = s + n #senal de entrada al ADC, senoidal + ruido analog
sQ = np.round(sR/q) * q #salida del ADC, cuantizacion
nq = sQ - sR #ruido cuantizacion

#%% ej a) 
#GRAFICO TEMPORAL

plt.figure(figsize=(12,6))

plt.plot(tt, sQ,label='sQ (ADC out)')

plt.plot(tt, sR,linestyle=':',marker='o',markersize=2,color='green',label='sR = s + n (ADC in)')

plt.plot(tt, s,linestyle=':',label='s (analog)', color='yellow')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')

plt.title( f'Señal muestreada por un ADC de {B} bits - '
    f'±VR = {vfs} V - q = {q:.3f} V  - kn = {kn} ')

plt.grid()
plt.legend()
plt.show()

#RUIDO DE CUANTIZACION EN EL TIEMPO - esto no va en el tp

plt.figure(figsize=(10,5))

plt.plot(tt, nq,linestyle='None',marker='.',markersize=3)

plt.axhline(q/2, color='red', linestyle='--', label='+q/2')
plt.axhline(-q/2, color='red', linestyle='--', label='-q/2')

plt.xlabel('Tiempo [s]')
plt.ylabel('Error de cuantización [V]')
plt.title('Ruido de cuantización nq = sQ - sR')

plt.grid()
plt.legend()
plt.show()

#%% HISTOGRAMA DEL RUIDO DE CUANTIZACION

plt.figure(figsize=(10,5))

plt.hist(nq,bins=10,edgecolor='black')

plt.axvline(-q/2,color='red',linestyle='--',label='-q/2')

plt.axvline(q/2,color='red',linestyle='--',label='+q/2')

plt.xlabel('Error de cuantización [V]')
plt.ylabel('Cantidad de muestras')

plt.title(f'Ruido de cuantización para {B} bits - '
    f'±VR = {vfs} V - q = {q:.3f}-V -  - kn = {kn}')

plt.grid()
plt.legend()
plt.show()

#%% MEDIA Y VARIANZA DEL RUIDO DE CUANTIZACION
#condiciones necesarias pero no suficientes para que sea uniforme
media_nq = np.mean(nq)
var_nq = np.var(nq)

var_teorica = q**2 / 12

print("Media nq =", media_nq)
print("Varianza medida nq =", var_nq)
print("Varianza teórica q^2/12 =", var_teorica)

# Kolmogorov-Smirnov test

# límites teóricos de la distribución uniforme
a = -q/2
b = q/2

resultado = stats.kstest(nq,'uniform',args=(a, b-a))

alpha = 0.05

print("p-value =", resultado.pvalue)

if resultado.pvalue > alpha:
    print("La señal es compatible con una distribución uniforme.")
else:
    print("La señal NO es compatible con una distribución uniforme.")
#%% AUTOCORRELACION DEL RUIDO DE CUANTIZACION

correlacion_nq = np.correlate(nq, nq, mode='full') / N

retardos = np.arange(-N+1, N)#para poner eje x en funcion de k y leer mejor la autocorrelacion

plt.figure(figsize=(10,5))

plt.plot(retardos,correlacion_nq,
         linestyle='None',
         marker='.',
         markersize=3)

plt.xlabel('Retardo [muestras]')
plt.ylabel('Autocorrelación')
plt.title('Autocorrelación del ruido de cuantización')

plt.grid()
plt.show()

#%% FFT PARA ANALISIS ESPECTRAL

SQ = np.fft.fft(sQ)
N_ANALOG = np.fft.fft(n)
NQ = np.fft.fft(nq)
freq = np.arange(N//2) * fs/N#creamos eje de frecuencias genera 0,1,2...499
#calc potencia
PSD_sQ = 2 * np.abs(SQ[:N//2]/N)**2
PSD_n = 2 * np.abs(N_ANALOG[:N//2]/N)**2
PSD_nq = 2 * np.abs(NQ[:N//2]/N)**2
#pasamos a db
PSD_sQ_db = 10*np.log10(PSD_sQ)
PSD_n_db = 10*np.log10(PSD_n)
PSD_nq_db = 10*np.log10(PSD_nq)
#calculamos pisos de ruido
piso_n = 10*np.log10(np.mean(PSD_n))
piso_nq = 10*np.log10(np.mean(PSD_nq))
piso_sQ= 10*np.log10(np.mean(PSD_sQ))

print("Piso ruido analógico =", piso_n, "dB")
print("Piso ruido cuantización =", piso_nq, "dB")



#%% GRAFICO ESPECTRAL FINAL

plt.figure(figsize=(12, 6))

# Señal principal (azul eléctrico intenso y línea más gruesa para resaltar)
plt.plot(freq, 
         PSD_sQ_db, 
         color='#0052CC', 
         linewidth=1.8, 
         label='ADC out')

# Espectros de ruido (colores diferenciados y semitransparentes)
plt.plot(freq, 
         PSD_n_db, 
         color='#7A869A', 
         linestyle=':', 
         linewidth=1.5, 
         label='n')

plt.plot(freq, 
         PSD_nq_db, 
         color='#FFAB00', 
         linestyle=':', 
         linewidth=1.5, 
         label='nq')

# Pisos de ruido (líneas punteadas en tonos vivos alineados a sus referencias)
plt.axhline(piso_n, 
            color='#FF5630', 
            linestyle='--', 
            linewidth=1.5, 
            label=f'Piso n = {piso_n:.1f} dB')

plt.axhline(piso_nq, 
            color='#36B37E', 
            linestyle='--', 
            linewidth=1.5, 
            label=f'Piso nq = {piso_nq:.1f} dB')

plt.xlabel('Frecuencia [Hz]', fontsize=11, fontweight='bold')
plt.ylabel('Potencia por bin [dB]', fontsize=11, fontweight='bold')

plt.title(
    f'Señal muestreada por un ADC de {B} bits - '
    f'±VR = {vfs} V - q = {q:.3f} V',
    fontsize=12,
    fontweight='bold',
    pad=12
)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()

#%% BONUS - SNR EN FUNCION DE LA CANTIDAD DE BITS

B_valores = [4, 8, 16]# valores de B que queremos comparar
sigma_x = np.std(s)
VFS = 2 * vfs
kn_bonus = 1 # mantenemos kn = 1 para comparar solamente el efecto de B
resultados = []

for B_aux in B_valores:
    q_aux = (2 * vfs) / (2**B_aux)
    Pq_aux = q_aux**2 / 12# potencia teorica del ruido de cuantizacion
    Pn_aux = kn_bonus * Pq_aux  # potencia del ruido analogico
    sigma_n_aux = np.sqrt(Pn_aux)# genero el ruido analogico
    n_aux = np.random.normal(0, sigma_n_aux, N)
    sR_aux = s + n_aux
    sQ_aux = np.round(sR_aux / q_aux) * q_aux
    nq_aux = sQ_aux - sR_aux    # entrada y salida del ADC

    # SNR TEORICO SEGUN HOLTON

    SNR_teorico = (6.02 * B_aux+ 10.79+ 20 * np.log10(sigma_x / VFS))

    # SNR MEDIDO DE CUANTIZACION
    # potencia señal / potencia error de cuantizacion
    SNR_cuant_medido = 10 * np.log10(np.var(s) / np.var(nq_aux))

    # SNR TOTAL DE LA DIGITALIZACION
    # compara la salida del ADC con la senoidal ideal incluye ruido analogico + cuantizacion

    error_total = sQ_aux - s

    SNR_digital = 10 * np.log10( np.var(s) / np.var(error_total))

    # guardamos resultados
    resultados.append([B_aux,q_aux,SNR_teorico,SNR_cuant_medido,SNR_digital])


#%% TABLA COMPARATIVA

print()
print("TABLA COMPARATIVA DE SNR")
print()

print(
    f"{'B [bits]':<10}"
    f"{'q [V]':<15}"
    f"{'SNR Holton [dB]':<20}"
    f"{'SNR cuant. [dB]':<20}"
    f"{'SNR total [dB]':<20}"
)

print("-"*85)

for fila in resultados:
    B_aux, q_aux, snr_teo, snr_cuant, snr_total = fila
    print(
        f"{B_aux:<10}"
        f"{q_aux:<15.6e}"
        f"{snr_teo:<20.2f}"
        f"{snr_cuant:<20.2f}"
        f"{snr_total:<20.2f}")


#codigos con grafs SUBPLOT en una misma fig para cada k 
#%% CASO kn = 0.1 - modificamos variables dependientes de kn

kn = 0.1

Pn = kn * Pq
sigma_n = np.sqrt(Pn)

n = np.random.normal(0, sigma_n, N)

sR = s + n
sQ = np.round(sR/q) * q
nq = sQ - sR


#%% TEST KOLMOGOROV-SMIRNOV

a = -q/2
b = q/2

resultado = stats.kstest(
    nq,
    'uniform',
    args=(a, b-a)
)

alpha = 0.05

print("p-value =", resultado.pvalue)

if resultado.pvalue > alpha:
    print("La señal es compatible con una distribución uniforme.")
else:
    print("La señal NO es compatible con una distribución uniforme.")


#%% AUTOCORRELACION

correlacion_nq = np.correlate(
    nq,
    nq,
    mode='full'
) / N

retardos = np.arange(-N+1, N)


#%% ANALISIS ESPECTRAL

SQ = np.fft.fft(sQ)
N_ANALOG = np.fft.fft(n)
NQ = np.fft.fft(nq)

freq = np.arange(N//2) * fs/N

# potencia por bin
PSD_sQ = 2 * np.abs(SQ[:N//2]/N)**2
PSD_n = 2 * np.abs(N_ANALOG[:N//2]/N)**2
PSD_nq = 2 * np.abs(NQ[:N//2]/N)**2

# paso a dB
PSD_sQ_db = 10*np.log10(PSD_sQ + 1e-20)
PSD_n_db = 10*np.log10(PSD_n + 1e-20)
PSD_nq_db = 10*np.log10(PSD_nq + 1e-20)

# pisos de ruido
piso_n = 10*np.log10(np.mean(PSD_n))
piso_nq = 10*np.log10(np.mean(PSD_nq))

print("Piso ruido analógico =", piso_n, "dB")
print("Piso ruido cuantización =", piso_nq, "dB")


#%% FIGURA RESUMEN

fig, axs = plt.subplots(2, 2, figsize=(15, 10))


# --------------------------------------------------
# 1) GRAFICO TEMPORAL
# --------------------------------------------------

axs[0,0].plot(
    tt,
    sQ,
    label='sQ (ADC out)'
)

axs[0,0].plot(
    tt,
    sR,
    linestyle=':',
    marker='o',
    markersize=2,
    color='green',
    label='sR = s + n (ADC in)'
)

axs[0,0].plot(
    tt,
    s,
    linestyle=':',
    color='yellow',
    label='s (analog)'
)

axs[0,0].set_xlabel('Tiempo [s]')
axs[0,0].set_ylabel('Amplitud [V]')

axs[0,0].set_title(
    'Señales en el dominio temporal'
)

axs[0,0].grid()
axs[0,0].legend()


# --------------------------------------------------
# 2) HISTOGRAMA DEL RUIDO DE CUANTIZACION
# --------------------------------------------------

axs[0,1].hist(
    nq,
    bins=10,
    edgecolor='black'
)

axs[0,1].axvline(
    -q/2,
    color='red',
    linestyle='--',
    label='-q/2'
)

axs[0,1].axvline(
    q/2,
    color='red',
    linestyle='--',
    label='+q/2'
)

axs[0,1].set_xlabel('Error de cuantización [V]')
axs[0,1].set_ylabel('Cantidad de muestras')

axs[0,1].set_title(
    'Histograma de $n_q$'
)

axs[0,1].grid()
axs[0,1].legend()


# --------------------------------------------------
# 3) AUTOCORRELACION
# --------------------------------------------------

axs[1,0].plot(
    retardos,
    correlacion_nq,
    linestyle='None',
    marker='.',
    markersize=3
)

axs[1,0].set_xlabel('Retardo [muestras]')
axs[1,0].set_ylabel('Autocorrelación')

axs[1,0].set_title(
    'Autocorrelación de $n_q$'
)

axs[1,0].grid()


# --------------------------------------------------
# 4) ESPECTRO
# --------------------------------------------------

axs[1,1].plot(
    freq,
    PSD_sQ_db,
    color='#0052CC',
    linewidth=1.5,
    label='ADC out'
)

axs[1,1].plot(
    freq,
    PSD_n_db,
    color='#7A869A',
    linestyle=':',
    linewidth=1.2,
    label='n'
)

axs[1,1].plot(
    freq,
    PSD_nq_db,
    color='#FFAB00',
    linestyle=':',
    linewidth=1.2,
    label='nq'
)

axs[1,1].axhline(
    piso_n,
    color='#FF5630',
    linestyle='--',
    linewidth=1.2,
    label=f'Piso n = {piso_n:.1f} dB'
)

axs[1,1].axhline(
    piso_nq,
    color='#36B37E',
    linestyle='--',
    linewidth=1.2,
    label=f'Piso nq = {piso_nq:.1f} dB'
)

axs[1,1].set_xlabel('Frecuencia [Hz]')
axs[1,1].set_ylabel('Potencia por bin [dB]')

axs[1,1].set_title(
    'Análisis espectral'
)

axs[1,1].grid(True, linestyle='--', alpha=0.5)
axs[1,1].legend(fontsize=8)


# titulo general
fig.suptitle(
    f'ADC de {B} bits - ±VR = {vfs} V - '
    f'q = {q:.3f} V - kn = {kn}',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.show()