# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:09:56 2026

@author: hoffe
"""

# -*- coding: utf-8 -*-
"""
TS8 - FILTRADO NO LINEAL
Analisis y Procesamiento de Señales
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
from scipy.interpolate import CubicSpline

plt.close('all')

#%% CARGA DE DATOS
# =============================================================================

# Cargar el archivo ECG
mat = sio.loadmat('ECG_TP4.mat')

s = mat['ecg_lead'].flatten()
fs = 1000  # Frecuencia de muestreo [Hz]
cant_muestras = len(s)
t = np.arange(cant_muestras) / fs

# Extraer detecciones reales provistas por la cátedra
qrs = mat['qrs_detections'].flatten().astype(int)


#%% 1) FILTRO DE MEDIANA


# Duración de las ventanas
t1 = 0.200  # 200 ms
t2 = 0.600  # 600 ms

# Conversión a cantidad de muestras y asegurar longitud impar
N1 = int(t1 * fs)
N1 = N1 + 1 if N1 % 2 == 0 else N1

N2 = int(t2 * fs)
N2 = N2 + 1 if N2 % 2 == 0 else N2

# Primera mediana: 200 ms (elimina picos QRS)
m_global = signal.medfilt(s, kernel_size=N1)

# Segunda mediana: 600 ms (extrae la componente lenta)
b_est_mediana = signal.medfilt(m_global, kernel_size=N2)

# ECG corregido global
ecg_mediana = s - b_est_mediana


#%% GRÁFICOS: FILTRO DE MEDIANA POR REGIONES
# =============================================================================

# Margen para evitar efectos de borde al filtrar por tramos
margen = N1//2 + N2//2


# Regiones con ruido (en muestras)
regs_ruido = (
    [4000, 5500],
    [10000, 11000],
    [100 * fs, 200 * fs],
)

# Regiones sin ruido (minutos a muestras)
regs_sin_ruido = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

regiones = {
    'Con ruido': regs_ruido,
    'Sin ruido': regs_sin_ruido
}

#%% Filtrado y gráficos

for tipo, regs_interes in regiones.items():

    for ii in regs_interes:

        inicio = int(max(0, ii[0]))
        fin = int(min(cant_muestras, ii[1]))

        if fin <= inicio:
            continue

        # Ampliamos la región para aplicar las medianas
        inicio_ext = max(0, inicio - margen)
        fin_ext = min(cant_muestras, fin + margen)

        s_region = s[inicio_ext:fin_ext]

        # Primera mediana: 200 ms
        med_200 = signal.medfilt(s_region, kernel_size=N1)

        # Segunda mediana: 600 ms
        b_est = signal.medfilt(med_200, kernel_size=N2)

        # Recortamos a la región de interés
        i0 = inicio - inicio_ext
        i1 = fin - inicio_ext

        ecg_original = s_region[i0:i1]
        mediana_200 = med_200[i0:i1]
        linea_base = b_est[i0:i1]

        # ECG corregido
        ecg_filtrado = ecg_original - linea_base

        zoom_region = np.arange(inicio, fin)

        # Gráficos
        fig, ax = plt.subplots(3, 1, figsize=(12, 8),
                               sharex=True)

        # Primera mediana
        ax[0].plot(zoom_region, ecg_original,
                   label='ECG original', alpha=0.7)

        ax[0].plot(zoom_region, mediana_200,
                   label='Mediana 200 ms')

        ax[0].set_title('Primera mediana - 200 ms')
        ax[0].legend()
        ax[0].grid()

        # Estimación de línea de base
        ax[1].plot(zoom_region, ecg_original,
                   label='ECG original', alpha=0.6)

        ax[1].plot(zoom_region, linea_base,
                   label='Línea de base estimada',
                   color='red', linewidth=2)

        ax[1].set_title('Estimación de línea de base - 600 ms')
        ax[1].legend()
        ax[1].grid()

        # ECG corregido
        ax[2].plot(zoom_region, ecg_original,
                   label='ECG original', alpha=0.5)

        ax[2].plot(zoom_region, ecg_filtrado,
                   label='ECG corregido', color='blue')

        ax[2].set_title('ECG después de restar la línea de base')
        ax[2].set_xlabel('Muestras (#)')
        ax[2].legend()
        ax[2].grid()

        plt.suptitle(
            f'Filtro de mediana - {tipo}: '
            f'muestras {inicio} a {fin}'
        )

        plt.tight_layout()
        plt.show()
        
from scipy.interpolate import CubicSpline

# ============================================================
# 1. DETECCIONES QRS Y PUNTOS ISOELECTRICOS
# ============================================================

# Detecciones QRS provistas en el archivo
qrs = mat['qrs_detections'].flatten().astype(int)
qrs = np.unique(qrs[(qrs >= 0) & (qrs < len(s))])

print("Cantidad de QRS detectados:", len(qrs))

# Desplazamiento respecto de cada QRS (80 ms)
n0 = int(0.08 * fs)

# Puntos de referencia para estimar la linea de base
m = qrs - n0
m = np.unique(m[(m >= 0) & (m < len(s))])

# Vector temporal
t = np.arange(len(s)) / fs

# ============================================================
# 2. ECG COMPLETO CON QRS Y PUNTOS ISOELECTRICOS
# ============================================================

plt.figure(figsize=(16, 5))
plt.plot(t, s, label='ECG original', linewidth=0.8, alpha=0.8)
plt.scatter(qrs / fs, s[qrs], color='red', marker='x', s=12, label='Detecciones QRS', zorder=3)
plt.scatter(m / fs, s[m], color='green', marker='o', s=12, label='Puntos isoeléctricos - 80 ms', zorder=4)

plt.title('ECG completo - QRS y puntos isoeléctricos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
# ============================================================
# 3. ZOOM PARA VERIFICAR LOS PUNTOS ISOELECTRICOS
# ============================================================

# Regiones para inspeccionar la ubicacion de los puntos
regs_zoom = (
    [100 * fs, 105 * fs],
    [700500, 703000],  # Zona añadida para evaluar ectópicos
)

for ii in regs_zoom:
    inicio = int(max(0, ii[0]))
    fin = int(min(len(s), ii[1]))

    if fin <= inicio: continue

    zoom_region = np.arange(inicio, fin)
    qrs_region = qrs[(qrs >= inicio) & (qrs < fin)]
    m_region = m[(m >= inicio) & (m < fin)]

    plt.figure(figsize=(13, 5))
    plt.plot(zoom_region, s[zoom_region], label='ECG original', linewidth=1.5)
    plt.scatter(qrs_region, s[qrs_region], color='red', marker='x', s=80, label='Detecciones QRS', zorder=4)
    plt.scatter(m_region, s[m_region], color='green', marker='o', s=65, label='Puntos isoeléctricos', zorder=5)

    plt.title(f'Puntos isoeléctricos - muestras {inicio} a {fin}')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 4. INTERPOLACION MEDIANTE SPLINES CUBICOS
# ============================================================

valores = s[m]
spline = CubicSpline(m, valores, bc_type='natural')
n = np.arange(len(s))
b_spline = spline(np.clip(n, m[0], m[-1]))
ecg_spline = s - b_spline

# ============================================================
# 5. GRAFICOS - SPLINE CUBICO (100 A 105 SEGUNDOS)
# ============================================================

inicio = int(100 * fs)
fin = int(105 * fs)
zoom_region = np.arange(inicio, fin)
m_region = m[(m >= inicio) & (m < fin)]

fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

ax[0].plot(zoom_region, s[zoom_region], label='ECG original')
ax[0].plot(zoom_region, b_spline[zoom_region], label='Línea de base - Spline', color='red', linewidth=2)
ax[0].scatter(m_region, s[m_region], color='green', s=45, label='Puntos isoeléctricos', zorder=5)
ax[0].set_title('Estimación de línea de base con spline cúbico')
ax[0].legend()
ax[0].grid()

ax[1].plot(zoom_region, s[zoom_region], label='ECG original', alpha=0.5)
ax[1].plot(zoom_region, ecg_spline[zoom_region], label='ECG corregido', color='blue')
ax[1].set_title('ECG después de restar la línea de base')
ax[1].set_xlabel('Muestras (#)')
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# ============================================================
# 6. REGIONES DE INTERES - SPLINES CUBICOS
# ============================================================

cant_muestras = len(s)

regs_ruido = (
    [4000, 5500],
    [10000, 11000],
    [100 * fs, 200 * fs],
)

regs_sin_ruido = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

regiones = {
    'Con ruido': regs_ruido,
    'Sin ruido': regs_sin_ruido
}

for tipo, regs_interes in regiones.items():
    for ii in regs_interes:
        inicio = int(max(0, ii[0]))
        fin = int(min(cant_muestras, ii[1]))

        if fin <= inicio: continue
        zoom_region = np.arange(inicio, fin)
        m_region = m[(m >= inicio) & (m < fin)]

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        ax[0].plot(zoom_region, s[zoom_region], label='ECG original', alpha=0.7)
        ax[0].plot(zoom_region, b_spline[zoom_region], label='Línea de base - Spline', color='red', linewidth=2)
        ax[0].scatter(m_region, s[m_region], color='green', s=35, label='Puntos isoeléctricos', zorder=5)
        ax[0].set_title('Estimación de línea de base')
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(zoom_region, s[zoom_region], label='ECG original', alpha=0.5)
        ax[1].plot(zoom_region, ecg_spline[zoom_region], label='ECG corregido', color='blue')
        ax[1].set_title('ECG después de restar la línea de base')
        ax[1].set_xlabel('Muestras (#)')
        ax[1].legend()
        ax[1].grid()

        plt.suptitle(f'Spline cúbico - {tipo}: muestras {inicio} a {fin}')
        plt.tight_layout()
        plt.show()

#%% 3) FILTRO ADAPTADO - MATCHED FILTER
# =============================================================================

# 1. PATRON QRS Y DETECCIONES DE REFERENCIA
patron = mat['qrs_pattern1'].flatten().astype(float)
qrs_ref = mat['qrs_detections'].flatten().astype(int)
qrs_ref = np.unique(qrs_ref[(qrs_ref >= 0) & (qrs_ref < len(s))])

print('Longitud del patron:', len(patron), 'muestras')
print('Latidos de referencia:', len(qrs_ref))

plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(patron)), patron, color='C0')
plt.title('Patron QRS provisto por la catedra')
plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. PREPROCESAMIENTO
# Filtro pasabajos Butterworth de orden 2
# Frecuencia de corte: 15 Hz
sos = signal.butter(2, 15, btype='lowpass',
                    fs=fs, output='sos')

ecg_pre = signal.sosfiltfilt(sos, s.astype(float))
# Normalizacion: media cero y desviacion estandar unitaria.
std_ecg = np.std(ecg_pre)
std_patron = np.std(patron)
if std_ecg == 0 or std_patron == 0:
    raise ValueError('La señal ECG o el patron QRS tiene desviacion estandar nula.')

ecg_norm = (ecg_pre - np.mean(ecg_pre)) / std_ecg
patron_norm = (patron - np.mean(patron)) / std_patron

# 3. FILTRO ADAPTADO = CORRELACION CON LA PLANTILLA
salida_fa = signal.correlate(ecg_norm, patron_norm,
                            mode='same', method='fft')
salida_fa = salida_fa / len(patron_norm)  # Escala conveniente

# Para saber donde comienza la plantilla en la señal cuando hay coincidencia:
lags = signal.correlation_lags(len(ecg_norm), len(patron_norm),
                               mode='same')

# 4. SEÑAL DE DETECCION: valor absoluto de la correlacion
deteccion = np.abs(salida_fa)

# 5. DETECCION DE PICOS (valores iniciales para ajustar)
distancia_min = int(0.30 * fs)  # 300 ms entre detecciones
prominencia_min = 1.0 * np.std(deteccion)

peaks, propiedades = signal.find_peaks(
    deteccion,
    distance=distancia_min,
    prominence=prominencia_min
)

# 6. ALINEACION TEMPORAL
# El maximo de correlacion corresponde al inicio de la plantilla (lag)
# mas la posicion del pico dominante dentro de la plantilla.
pos_qrs_patron = int(np.argmax(np.abs(patron_norm)))
qrs_detectados = lags[peaks] + pos_qrs_patron
qrs_detectados = np.unique(qrs_detectados[
    (qrs_detectados >= 0) & (qrs_detectados < len(s))
])

print('Separacion minima:', distancia_min, 'muestras')
print('Prominencia elegida:', round(prominencia_min, 3))
print('Indice del pico dominante del patron:', pos_qrs_patron)
print('Latidos detectados:', len(qrs_detectados))

#%% GRAFICOS DEL FILTRO ADAPTADO - ECG COMPLETO Y ZOOM
# =============================================================================

# Normalizacion SOLO para representar las dos curvas en un mismo eje
escala_ecg = max(float(np.max(np.abs(s))), 1e-12)
escala_det = max(float(np.max(deteccion)), 1e-12)
ecg_graf = s / escala_ecg
deteccion_graf = deteccion / escala_det

t = np.arange(len(s)) / fs

# Grafico 1: ECG completo
plt.figure(figsize=(16, 6))
plt.plot(t, ecg_graf, color='blue', linewidth=0.8,
         alpha=0.7, label='ECG original (normalizado)')
plt.plot(t, deteccion_graf, color='orange', linewidth=1,
         alpha=0.75, label='|Correlacion| (normalizada)')
plt.scatter(qrs_ref / fs, ecg_graf[qrs_ref],
            color='red', marker='x', s=15, zorder=4,
            label='QRS de referencia')
plt.scatter(qrs_detectados / fs, ecg_graf[qrs_detectados],
            edgecolors='green', facecolors='none', marker='o',
            s=22, zorder=5, label='QRS detectados')
plt.title('ECG completo - Filtro adaptado y deteccion QRS')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud normalizada (escalas independientes)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Grafico 2: acercamiento entre 326 y 338 segundos
inicio = max(0, int(326 * fs))
fin = min(len(s), int(338 * fs))
zoom = np.arange(inicio, fin)

qrs_ref_zoom = qrs_ref[(qrs_ref >= inicio) & (qrs_ref < fin)]
qrs_det_zoom = qrs_detectados[
    (qrs_detectados >= inicio) & (qrs_detectados < fin)
]

if len(zoom) > 0:
    plt.figure(figsize=(16, 6))
    plt.plot(t[zoom], ecg_graf[zoom], color='blue',
             linewidth=1.2, alpha=0.8,
             label='ECG original (normalizado)')
    plt.plot(t[zoom], deteccion_graf[zoom], color='orange',
             linewidth=1.2, label='|Correlacion| (normalizada)')
    plt.scatter(qrs_ref_zoom / fs, ecg_graf[qrs_ref_zoom],
                color='red', marker='x', s=65,
                label='QRS de referencia', zorder=4)
    plt.scatter(qrs_det_zoom / fs, ecg_graf[qrs_det_zoom],
                edgecolors='green', facecolors='none', marker='o',
                s=75, label='QRS detectados', zorder=5)
    plt.title('Filtro adaptado - Zoom entre 326 y 338 s')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud normalizada (escalas independientes)')
    plt.xlim(326, 338)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print('El registro no alcanza el intervalo 326-338 s.')


#%% METRICAS - COMPARACION CON DETECCIONES DE REFERENCIA
# =============================================================================

# Definimos una tolerancia en torno al QRS de referencia
tolerancia_ms = 80
tolerancia = int(tolerancia_ms * fs / 1000)

# Emparejamiento UNO A UNO de dos arreglos ordenados
# Evita contar varias detecciones como un mismo verdadero positivo
i = 0
j = 0
pares = []  # (muestra de referencia, muestra detectada)

while i < len(qrs_ref) and j < len(qrs_detectados):
    diferencia = int(qrs_detectados[j] - qrs_ref[i])

    if abs(diferencia) <= tolerancia:
        pares.append((int(qrs_ref[i]), int(qrs_detectados[j])))
        i += 1
        j += 1

    elif diferencia < -tolerancia:
        j += 1

    else:
        i += 1

VP = len(pares)
FN = len(qrs_ref) - VP
FP = len(qrs_detectados) - VP

sensibilidad = VP / (VP + FN) if (VP + FN) else 0.0
vpp = VP / (VP + FP) if (VP + FP) else 0.0
f1 = (2 * sensibilidad * vpp / (sensibilidad + vpp)
      if (sensibilidad + vpp) else 0.0)

print('\nRESULTADOS DEL DETECTOR QRS')
print('Referencias:', len(qrs_ref))
print('Detecciones:', len(qrs_detectados))
print('Tolerancia:', tolerancia_ms, 'ms')
print('VP:', VP, '| FP:', FP, '| FN:', FN)
print(f'Sensibilidad: {100*sensibilidad:.2f} %')
print(f'Valor predictivo positivo: {100*vpp:.2f} %')
print(f'F1-score: {100*f1:.2f} %')

# Analisis del desfase de las detecciones que coincidieron
if pares:
    desfasajes_ms = np.array([(det - ref) * 1000 / fs
                             for ref, det in pares])
    print(f'Desfase mediano (detectado - referencia): '
          f'{np.median(desfasajes_ms):.1f} ms')


# ============================================================
# POSICIONES DE VP, FP Y FN PARA GRAFICAR
# ============================================================

# Verdaderos positivos: detecciones correctamente emparejadas
vp_det = np.array([det for ref, det in pares], dtype=int)

# Falsos positivos: detecciones que no coinciden con referencias
fp_det = np.setdiff1d(qrs_detectados, vp_det)

# Falsos negativos: referencias sin deteccion correspondiente
vp_ref = np.array([ref for ref, det in pares], dtype=int)
fn_ref = np.setdiff1d(qrs_ref, vp_ref)

# Ubicacion de los errores en segundos
print('\nUBICACION DE LOS ERRORES')
print('FP en segundos:', fp_det / fs)
print('FN en segundos:', fn_ref / fs)


#%% GRAFICO COMPLETO - VP, FP Y FN

t = np.arange(len(s)) / fs

plt.figure(figsize=(16, 6))

# ECG original
plt.plot(t, s, color='blue', linewidth=0.8,
         alpha=0.7, label='ECG original')

# Verdaderos positivos
plt.scatter(vp_det / fs, s[vp_det],
            facecolors='none', edgecolors='green',
            marker='o', s=25, label='VP', zorder=5)

# Falsos positivos
plt.scatter(fp_det / fs, s[fp_det],
            color='red', marker='s',
            s=65, label='FP', zorder=6)

# Falsos negativos
plt.scatter(fn_ref / fs, s[fn_ref],
            color='magenta', marker='x',
            s=85, linewidths=2,
            label='FN', zorder=6)

plt.title('Evaluacion del detector QRS - Registro completo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()



#%% ZOOM - REGIONES CON ERRORES DE DETECCION

# Intervalos de interes en segundos
regiones_error = (
    [142.5, 144],
    [247.3, 249.2],
    [283.3, 284.9],
    [512.2, 513.8],
)

fig, ax = plt.subplots(2, 2, figsize=(16, 9))
ax = ax.flatten()

for k, (t_ini, t_fin) in enumerate(regiones_error):

    inicio = int(t_ini * fs)
    fin = int(t_fin * fs)

    zoom = np.arange(inicio, fin)

    # Detecciones dentro de cada region
    vp_zoom = vp_det[(vp_det >= inicio) & (vp_det < fin)]
    fp_zoom = fp_det[(fp_det >= inicio) & (fp_det < fin)]
    fn_zoom = fn_ref[(fn_ref >= inicio) & (fn_ref < fin)]

    # ECG original
    ax[k].plot(zoom / fs, s[zoom],
               color='blue', linewidth=1.2,
               alpha=0.8, label='ECG original')

    # Verdaderos positivos: circulos verdes
    ax[k].scatter(vp_zoom / fs, s[vp_zoom],
                  facecolors='none', edgecolors='green',
                  marker='o', s=65, linewidths=1.5,
                  label='VP', zorder=5)

    # Falsos positivos: cuadrados rojos
    ax[k].scatter(fp_zoom / fs, s[fp_zoom],
                  color='red', marker='s',
                  s=85, label='FP', zorder=6)

    # Falsos negativos: cruces violetas
    ax[k].scatter(fn_zoom / fs, s[fn_zoom],
                  color='magenta', marker='x',
                  s=100, linewidths=2,
                  label='FN', zorder=6)

    ax[k].set_title(
        f'Region {k+1}: {t_ini} a {t_fin} s',
        fontsize=12
    )

    ax[k].set_xlabel('Tiempo [s]')
    ax[k].set_ylabel('Amplitud')
    ax[k].set_xlim(t_ini, t_fin)
    ax[k].grid(True, alpha=0.4)

# Leyenda unica para toda la figura
handles, labels = ax[0].get_legend_handles_labels()

fig.legend(handles, labels,
           loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Evaluacion del detector QRS - Regiones con errores',
             fontsize=15)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.show()



#%% BONUS A - FILTRADO MULTIRATE

import time

# 1. PARAMETROS

D = 10                  # Factor de diezmado
fs_nueva = fs / D       # 100 Hz

# Region de interes: 100 a 110 segundos
inicio = int(100 * fs)
fin = int(110 * fs)

# Agregamos un margen para evitar efectos de borde
margen = int(2 * fs)

inicio_ext = max(0, inicio - margen)
fin_ext = min(len(s), fin + margen)

s_region = s[inicio_ext:fin_ext]

# Indices de la region original dentro del segmento ampliado
i0 = inicio - inicio_ext
i1 = fin - inicio_ext

# Ventanas del filtro original
N1 = 201
N2 = 601

# Ventanas equivalentes a 100 Hz
N1_red = 21
N2_red = 61

# 2. FILTRO DE MEDIANA ORIGINAL (1000 Hz)


t0 = time.perf_counter()

med_200 = signal.medfilt(s_region, kernel_size=N1)
b_original = signal.medfilt(med_200, kernel_size=N2)

tiempo_original = time.perf_counter() - t0


# 3. FILTRO DE MEDIANA MULTIRATE (100 Hz)


t0 = time.perf_counter()

# Diezmado por 10 con filtro antialiasing incorporado
s_reducida = signal.resample_poly(s_region, up=1, down=D)

# Aplicamos las dos medianas a 100 Hz
med_200_red = signal.medfilt(s_reducida, kernel_size=N1_red)
b_reducida = signal.medfilt(med_200_red, kernel_size=N2_red)

# Interpolamos nuevamente a 1000 Hz
b_multirate = signal.resample_poly(b_reducida, up=D, down=1)

# Ajustamos la longitud al registro original
b_multirate = b_multirate[:len(s_region)]

tiempo_multirate = time.perf_counter() - t0


# 4. RECORTAMOS Y OBTENEMOS LOS ECG CORREGIDOS


ecg_original = s_region[i0:i1]

b_original = b_original[i0:i1]
b_multirate = b_multirate[i0:i1]

ecg_corregido_original = ecg_original - b_original
ecg_corregido_multirate = ecg_original - b_multirate

t_region = np.arange(inicio, fin) / fs


# 5. COMPARACION NUMERICA


# Diferencia entre las dos estimaciones de linea de base
rmse = np.sqrt(np.mean((b_original - b_multirate)**2))

print("RESULTADOS MULTIRATE")
print("Frecuencia original:", fs, "Hz")
print("Frecuencia reducida:", fs_nueva, "Hz")

print(f"Tiempo mediana original: {tiempo_original:.4f} s")
print(f"Tiempo mediana multirate: {tiempo_multirate:.4f} s")

if tiempo_multirate > 0:
    print(f"Factor de aceleracion: "
          f"{tiempo_original / tiempo_multirate:.2f} veces")

print(f"RMSE entre estimaciones: {rmse:.2f}")


# 6. GRAFICO 1 - LINEA DE BASE


plt.figure(figsize=(13, 5))

plt.plot(t_region, ecg_original,
         label='ECG original', alpha=0.35)

plt.plot(t_region, b_original,
         label='Mediana original (1000 Hz)',
         color='red', linewidth=2)

plt.plot(t_region, b_multirate,
         label='Mediana multirate (100 Hz)',
         color='green', linestyle='--', linewidth=2)

plt.title('Bonus - Comparacion de estimaciones de linea de base')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# 7. GRAFICO 2 - ECG CORREGIDO


plt.figure(figsize=(13, 5))

plt.plot(t_region, ecg_corregido_original,
         label='ECG corregido - Original',
         color='blue', alpha=0.7)

plt.plot(t_region, ecg_corregido_multirate,
         label='ECG corregido - Multirate',
         color='orange', linestyle='--', alpha=0.8)

plt.title('Bonus - Comparacion de ECG corregidos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# 8. GRAFICO 3 - TIEMPOS DE PROCESAMIENTO

plt.figure(figsize=(7, 4))

metodos = ['Original (1000 Hz)', 'Multirate (100 Hz)']
tiempos = [tiempo_original, tiempo_multirate]

barras = plt.bar(metodos, tiempos)

for barra, valor in zip(barras, tiempos):
    plt.text(barra.get_x() + barra.get_width()/2,
             valor, f'{valor:.4f} s',
             ha='center', va='bottom')

plt.title('Comparacion del costo computacional')
plt.ylabel('Tiempo de procesamiento [s]')
plt.ylim(0, max(tiempos) * 1.2)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

