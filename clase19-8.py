# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:13:48 2026

@author: hoffe
"""
import numpy as np 
import matplotlib.pyplot as plt

#%% definciones
fs=1000 # frecuencia de muestreo - hertz
N=1000 #cantidad de muestras

#%% funciones

def mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    #vmax=amplitud. argumento que si no me lo pasas le pongo valor 1
    #dc= valor de continua, valor medio, si no me lo pasas tomo por defecto valor 0
    #ff=frecuencia
    #nn cantidad de muestras
    #ph=fase
    #tt=tiempo
    
    
    #creamos el vector de tiempo que comenza en 0, termina en nn/fs y avanza cada 1/fs. 
    #el tiempo va desde 0 a (nn-1 muestras)*Ts --> rango desde 0 a N-1*Ts
    tt=np.arange(0, nn/fs, 1/fs)
    #xx calculara el valor de la senal para cada uno de eses tiempos
    xx=dc + vmax * np.sin (2*np.pi*ff*tt +ph)
    
    # hasta acá
    return(tt, xx)

#%% GENERAMOS LA SENOIDAL DE 1 WATT
vmax=np.sqrt(2) #CON ESTO YA DEFINIMOS LA SENAL DE 1 WATT
dc=0
ff=4 #poner frec que querramos de la senoidal
#como puse frecuencia 3 mi periodo deberia medir 1/3, entonces en mis n muestras tendria que tener 3 ciclos compeltos
ph=0 #fase 

#cant demuestras y fs ya esta definida

tt, xx = mi_funcion_sen( vmax = vmax, dc = dc, ff = ff, ph=ph, nn = N, fs = fs)

plt.plot(tt, xx)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.title(f'Señal senoidal: vmax={vmax} V, dc={dc} V, ff={ff} Hz, ph={ph} rad')
plt.grid()
plt.show()

print (np.var(xx)) #para chequear que la senal sneoidal es de 1 watt. 

#%% GENERAMOS EL RUIDO

snr_db = 20

# potencia de la señal
pot_senal = np.var(xx)

# potencia necesaria del ruido para obtener la SNR deseada
pot_ruido = pot_senal / (10**(snr_db/10))

# desvío estándar del ruido --> DESVIO ESTANDAR ES LA RAIZ DE LA VARIANZA
sigma_ruido = np.sqrt(pot_ruido)

# ruido gaussiano de media cero
ruido = np.random.normal(0, sigma_ruido, N)
#LA FUNCION NP.RANDOM.NORMAL --> np.random.normal(media, desvio_estandar, cantidad_de_muestras)
print("Potencia señal:", np.var(xx))
print("Potencia ruido:", np.var(ruido))

#%% ahora sumamos la senoidal con el ruido

#agregar: armar el generador como una funcion aparte y lograr sumar ambas. 
