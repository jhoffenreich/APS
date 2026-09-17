# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:52:45 2026

@author: hoffe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%% definiciones
fs = 1000
N = 1000

#%% funciones

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):
    
    tt = np.arange(0, nn/fs, 1/fs)
    
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
    
    return tt, xx


def mi_funcion_ruido(pot_senal=1, snr_db=40, nn=N):
    
    # calculamos la potencia que debe tener el ruido
    pot_ruido = pot_senal / (10**(snr_db/10))
    
    # np.random.normal necesita desvio estandar,
    # por eso hacemos la raiz cuadrada de la potencia
    sigma = np.sqrt(pot_ruido)
    
    # generamos ruido gaussiano de media 0
    ruido = np.random.normal(0, sigma, nn)
    
    return ruido


#%% comienzo del script

vmax = np.sqrt(2)
#trabajamos ocn vmax 1 y con frec=4 nos da -500j
#vmax=1
dc = 0
ff = 4
ph = 0

# generamos la senoidal
tt, xx = mi_funcion_sen(
    vmax=vmax,
    dc=dc,
    ff=ff,
    ph=ph,
    nn=N,
    fs=fs
)

# potencia de la señal
pot_senal = np.var(xx)

print("Potencia de la señal:", pot_senal)

# generamos ruido para una SNR de 20 dB
ruido = mi_funcion_ruido(
    pot_senal=pot_senal,
    snr_db=40,#ruido comparable con un cuantizador de 8 bits, senal apenitas ruido
    nn=N
)

print("Potencia del ruido:", np.var(ruido))

# sumamos señal + ruido
xx_ruido = xx + ruido

# comprobamos la SNR obtenida
snr_obtenida = 10*np.log10(np.var(xx) / np.var(ruido))

print("SNR obtenida:", snr_obtenida, "dB")


#%% gráficos
plt.figure()
plt.plot(tt, xx)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Señal senoidal de 1 W')
plt.grid()
plt.show()

plt.figure()
plt.plot(tt, ruido)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Ruido gaussiano')
plt.grid()
plt.show()

plt.figure()
plt.plot(tt, xx_ruido)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title('Señal senoidal con ruido - SNR = 20 dB')
plt.grid()
plt.show()

#%% FFT DE LA SENAL DEL SENO sin ruido 
fft_sen=np.fft.fft(xx) #nos devuelve un vector de numeros complejos
print("resultado fft")
print(fft_sen)

#vamos a analizar en al pestanas d evariable slos que nos pone en color. en mi caos 3 y 997. es el que varia mucho con respecto a los otros. donde esta el delta?
#yo tengo la senoidal definida de una manera, pensar con como la define el profe con el deltaf y k. 
#cambiarla para que quede definida con ese argumento

#n=np.arange(N) 
#delta_f=  fs/N
#senal_sen=np.sin(k*(fs/N)*n)
#chequear esto x las dudas. 

#una vez que sabemos que funcion ale agregamso el ruido, deberiamos ver un espectro plano que vamoa  a ver la sem que viene. deberiamos ver algo que tiende a la planitud.

#AGREGAMOS CONSIGNAS DE LA CLASE 20 DE AGOSTO 

#agregamos un fector de frecuencias, desde 0 a 499 porque no interesa la mitad del espectro
k = np.arange(N//2)
delta_f = fs/N
freq = k * delta_f 
#cone sto tenemos la fomrula: fk=k*deltaf y ya tenemos el eje de frecuencias listo. 

#%% FFT
#calculamos la fft de senoidal que ya esta arriba, ruido y suma de ambas
fft_ruido=np.fft.fft(ruido)
fft_suma=np.fft.fft(xx_ruido)

fft_sen_norm = fft_sen / N
fft_ruido_norm = fft_ruido / N
fft_suma_norm = fft_suma / N


#%% CALCULAMOS EL MODULO SOLO DE LA MITAD . LA OTRA ES REDUNDANTE
#mod_sen = np.abs(fft_sen[:N//2])
#mod_ruido = np.abs(fft_ruido[:N//2])
#mod_suma = np.abs(fft_suma[:N//2])
mod_sen = np.sqrt(2) * np.abs(fft_sen_norm[:N//2])
mod_ruido = np.sqrt(2) * np.abs(fft_ruido_norm[:N//2])
mod_suma = np.sqrt(2) * np.abs(fft_suma_norm[:N//2])

mod_sen_db = 20*np.log10(mod_sen + 1e-20)
mod_ruido_db = 20*np.log10(mod_ruido + 1e-20)
mod_suma_db = 20*np.log10(mod_suma + 1e-20)

#para 1W=0db 
#potencia x bin

#%% CALCULAMOS LA FASE 
fase_sen = np.angle(fft_sen[:N//2])
fase_ruido = np.angle(fft_ruido[:N//2])
fase_suma = np.angle(fft_suma[:N//2])
#%% grafico modulos y fase
#mod_sen_db = 20*np.log10(mod_sen)
#PARA SENO 
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(freq, mod_suma_db)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo en db')
plt.title('Módulo FFT en db - Señal + ruido')
plt.grid()


#senoidal fase
plt.subplot(2, 1, 2)
plt.plot(freq, fase_suma)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.title('Fase FFT - Señal senoidal + ruido')
plt.grid()

plt.tight_layout()
plt.show()

# grafico modulo en db VS MOD FFT 


plt.figure()
plt.plot(freq, mod_sen_db, label='MOD SEN DB')
plt.plot(freq,mod_sen, label='MOD SEN')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.title('Módulo FFT en dB - Señal senoidal')
plt.grid()
plt.show()

#PARA RUIDO 
plt.figure()
plt.plot(freq, mod_ruido)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo FFT - Señal ruido')
plt.grid()
plt.show()

#ruido fase 
plt.figure()
plt.plot(freq, fase_ruido)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.title('Fase FFT - Señal ruido')
plt.grid()
plt.show()

#PARA SUMA
plt.figure()
plt.plot(freq, mod_suma)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo')
plt.title('Módulo FFT - Señal suma')
plt.grid()
plt.show()

#suma fase
plt.figure()
plt.plot(freq, fase_suma)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.title('Fase FFT - Señal suma')
plt.grid()
plt.show()



#%% bloque cuantizador 
#queremos pasarle al sistema la senoid=ruido
xr = xx_ruido #senal antes de cuantizar
vfs=1.65
B=8#bits
q = (2*vfs) / (2**B)


#paso al otro 
xq = np.round(xr/q) * q  #ya cuantizada
#tambien quieor calcular el ruido de cuantizacion
nq=xq-xr

plt.figure()
plt.plot(xr, label='Señal antes de cuantizar')
plt.plot(xq,label= 'Señal cuantizada')#version cuantizada
plt.title('Señal original y señal cuantizada')
plt.grid()
plt.legend()


plt.figure()
plt.plot(nq/q, linestyle= '--')#deberia estar en +-1.5
plt.xlabel('Número de muestra')
plt.ylabel('nq / q')
plt.title('Ruido de cuantización normaliz')
plt.grid()

#verificamos que la media =0 
media_nq = np.mean(nq)
print("Media del ruido de cuantización:", media_nq)

var_nq = np.var(nq)
var_teorica = q**2 / 12

print("Varianza medida:", var_nq)
print("Varianza teórica q^2/12:", var_teorica)

#%% AUTOCORRELACION DEL RUIDO DE CUANT
#comprobamos si nq es incorrelado. 
correlacion = np.correlate(nq, nq, 'full') / N

plt.figure()
plt.plot(correlacion, linestyle='None', marker='.')
plt.ylabel('Autocorrelación')
plt.title('Autocorrelación del ruido de cuant')
plt.grid()
plt.show()

#%% comprobamos si es uniforme
#Kolmogorov-Smirnov test
# limites teoricos de la distribucion uniforme
a = -q/2
b = q/2

resultado = stats.kstest(nq, 'uniform', args=(a, b-a))

alpha = 0.05

print("p-value =", resultado.pvalue)

if resultado.pvalue > alpha:
    print("La señal es compatible con una distribución uniforme.")
else:
    print("La señal NO es compatible con una distribución uniforme.")
    
    #%% clase 2/9

# parametros de la consigna
fs = 1000
N = 1000
B = 4
vfs = 2
kn = 1/100

# resolucion frecuencial
delta_f = fs/N

# frecuencia de la senoidal
f0 = delta_f       # = 1 Hz

# amplitud para potencia/varianza unitaria
vmax = np.sqrt(2)

# genero nuevamente la senoidal
tt, s = mi_funcion_sen(vmax=vmax,dc=0,ff=f0,ph=0,nn=N,fs=fs)

print("Potencia señal =", np.var(s))

q = (2*vfs) / (2**B)

# potencia del ruido de cuantizacion
Pq = q**2 / 12

# potencia del ruido analogico
Pn = kn * Pq


#generamos ruido analog gausinao
sigma_n = np.sqrt(Pn)

n = np.random.normal(loc=0,scale=sigma_n,size=N)

#entrada al ADC sr=s+n
sR = s + n
#salida del adc --> cuantizamos 
sQ = np.round(sR/q) * q
#calculamos el ruido de cuantizacion
nq = sQ - sR

#s senal analogica ideal
#n ruido analogico agregado 
#sr entrada al adc
#sq salida del adc
#nq ruido producido x cuantizacion 

#%% HISTOGRAMA RUIDO DE CUANTIZACION

plt.figure()

plt.hist(nq,bins=10)

plt.axvline(-q/2, color='red', linestyle='--')
plt.axvline(q/2, color='red', linestyle='--')

plt.xlabel('Error de cuantización [V]')
plt.ylabel('Cantidad de muestras')
plt.title(
    f'Ruido de cuantización para {B} bits - '
    f'±VR = {vfs} V - q = {q:.3f} V')

plt.grid()
plt.show()

#grafico espectral 
# FFT PARA GRAFICO DE ADC

SQ = np.fft.fft(sQ)
N_ANALOG = np.fft.fft(n)
NQ = np.fft.fft(nq)

# frecuencia solamente entre 0 y fs/2
freq = np.arange(N//2) * fs/N

#queremos representar pot: 
PSD_sQ = 2 * np.abs(SQ[:N//2]/N)**2
PSD_n = 2 * np.abs(N_ANALOG[:N//2]/N)**2
PSD_nq = 2 * np.abs(NQ[:N//2]/N)**2
#en db:
PSD_sQ_db = 10*np.log10(PSD_sQ + 1e-20)
PSD_n_db = 10*np.log10(PSD_n + 1e-20)
PSD_nq_db = 10*np.log10(PSD_nq + 1e-20)

#calculamos donde aprox esta el piso de cada ruido
piso_n = 10*np.log10(np.mean(PSD_n))
piso_nq = 10*np.log10(np.mean(PSD_nq))

print("Piso ruido analogico =", piso_n, "dB")
print("Piso ruido cuantizacion =", piso_nq, "dB")

# GRAFICO FINAL

plt.figure(figsize=(12, 6))

# ADC out
plt.plot(freq,PSD_sQ_db,label='ADC out')

# ruido analogico
plt.plot(freq,PSD_n_db,linestyle=':',label='n')

# ruido cuantizacion
plt.plot(freq,PSD_nq_db,linestyle=':', label='nq')

# pisos promedio
plt.axhline(piso_n,linestyle='--',label=f'Piso n = {piso_n:.1f} dB')

plt.axhline(piso_nq, color='red',linestyle='--',label=f'Piso nq = {piso_nq:.1f} dB')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')

plt.title(f'Señal muestreada por un ADC de {B} bits - '
    f'±VR = {vfs} V - q = {q:.3f} V')

plt.grid()
plt.legend()
plt.show()


#%% AUTOCORRELACION DEL RUIDO DE CUANTIZACION

correlacion_nq = np.correlate(nq, nq, mode='full') / N

plt.figure()

plt.plot(correlacion_nq,linestyle='None',marker='.')

plt.xlabel('Retardo [muestras]')
plt.ylabel('Autocorrelación')
plt.title('Autocorrelación del ruido de cuantización')
plt.grid()
plt.show()


#%% GRAFICO TEMPORAL - ENTRADA Y SALIDA DEL ADC

plt.figure(figsize=(12, 6))

# salida cuantizada del ADC
plt.plot(tt,sQ,label='sQ = ADC out')

# entrada al ADC: señal + ruido
plt.plot(tt,sR,linestyle=':',marker='o',markersize=2,label='sR = s + n  (ADC in)')

# señal analógica ideal
#plt.plot(tt,s,linestyle=':',label='s (analog)')

plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')

plt.title( f'Señal muestreada por un ADC de {B} bits - '
    f'±VR = {vfs} V - q = {q:.3f} V')

plt.grid()
plt.legend()
plt.show()

#%% ERROR / RUIDO DE CUANTIZACION

plt.figure(figsize=(10,5))
plt.plot(tt, nq, linestyle='None', marker='.', markersize=2)

plt.xlabel('tiempo [s]')
plt.ylabel('Error de cuantización [V]')
plt.title('Ruido de cuantización nq = sQ - sR')
plt.grid()
plt.show()

#vemos pico alto en 0 y en demas chicos --> incorelado