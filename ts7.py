# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
plt.close('all')

#%% Plantilla de diseño
fs_ecg = 1000
nyq_frec = fs_ecg/2 # Frecuencia de Nyquist (500 Hz)
ripple = 1  #Atenuacion max permitida en la banda de paso, en dB
atenuacion = 40 #Atenuacion minima exigida en las bandas de rechazo, en dB

#Frecs que delimitan las bandas de paso, rechazo y de transicion. En Hz
ws1 = 0.1 #fin de la banda de rechazo inferior
wp1 = 1 #inicio de la banda d epaso
wp2 = 35 #fin de la banda d epaso
ws2 = 50 #comienzo de la banda de rechazo sup. 

# Plantilla normalizada a Nyquist (frecs entre 0 y 1)
frecs = np.array([0, ws1, wp1, wp2, ws2, nyq_frec])/nyq_frec
#limites de atenuacion de la plantilla expresados en db
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
#conversion de db a ganacia lineal de amplitud 
gains = 10**(gains_db/20)

#%% Diseño IIR - Butterworth
wp = [wp1, wp2]  # Límites de la banda de paso 
ws = [ws1, ws2]  # Límites de las bandas de rechazo 

sos_butt = signal.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='butter', fs=fs_ecg, output='sos')
#gpass: max atenuacion permitida en la BP (1dB)
#gstop:minima atenuacion requerida en las BR (40dB)
#output='sos' nos devuelve el filtro expresado en secciones de segundo orden. 
print('\nFILTRO BUTTERWORTH')
print('Secciones de segundo orden:', len(sos_butt))
print('Orden del filtro digital:', 2*len(sos_butt))

#ahora verificamos la rta en frec. calculamos su rta H(e^jomega) y graficamos su modulo en dB
f = np.unique(np.r_[np.linspace(0, 3, 3001), np.linspace(3, 100, 5000), np.linspace(100, 500, 3000), ws1, wp1, wp2, ws2])
W = 2*np.pi*f/fs_ecg
_, H_butt = signal.freqz_sos(sos_butt, worN=W)
mod_butt = 20*np.log10(np.maximum(np.abs(H_butt), 1e-12))

#verificacion numerica de la plantilla:
paso = (f>=wp1)&(f<=wp2)
rechazo_inf = f<=ws1
rechazo_sup = f>=ws2

print('Mínimo en banda de paso:', round(np.min(mod_butt[paso]), 2), 'dB')
print('Máximo en rechazo inferior:', round(np.max(mod_butt[rechazo_inf]), 2), 'dB')
print('Máximo en rechazo superior:', round(np.max(mod_butt[rechazo_sup]), 2), 'dB')


#%% Diseño IIR - Cauer
sos_cauer = signal.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='ellip', fs=fs_ecg, output='sos')

print('\nFILTRO CAUER')
print('Secciones de segundo orden:', len(sos_cauer))
print('Orden del filtro digital:', 2*len(sos_cauer))

# Respuesta en frecuencia
_, H_cauer = signal.freqz_sos(sos_cauer, worN=W)
mod_cauer = 20*np.log10(np.maximum(np.abs(H_cauer), 1e-12))

# Verificación de la plantilla
print('Mínimo en banda de paso:', round(np.min(mod_cauer[paso]), 2), 'dB')
print('Máximo en rechazo inferior:', round(np.max(mod_cauer[rechazo_inf]), 2), 'dB')
print('Máximo en rechazo superior:', round(np.max(mod_cauer[rechazo_sup]), 2), 'dB')


#%% Diseño IIR - Chebyshev II
sos_cheby2 = signal.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='cheby2', fs=fs_ecg, output='sos')

print('\nFILTRO CHEBYSHEV II')
print('Secciones de segundo orden:', len(sos_cheby2))
print('Orden del filtro digital:', 2*len(sos_cheby2))

# Respuesta en frecuencia
_, H_cheby2 = signal.freqz_sos(sos_cheby2, worN=W)
mod_cheby2 = 20*np.log10(np.maximum(np.abs(H_cheby2), 1e-12))

# Verificación de la plantilla
print('Mínimo en banda de paso:', round(np.min(mod_cheby2[paso]), 2), 'dB')
print('Máximo en rechazo inferior:', round(np.max(mod_cheby2[rechazo_inf]), 2), 'dB')
print('Máximo en rechazo superior:', round(np.max(mod_cheby2[rechazo_sup]), 2), 'dB')


#%% Comparación IIR - Butterworth, Chebyshev II y Cauer
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, limites in zip(ax, [(0, 100), (0, 3)]):
    a.plot(f, mod_butt, label='Butterworth')
    a.plot(f, mod_cheby2, label='Chebyshev II')
    a.plot(f, mod_cauer, label='Cauer')
    a.fill_between(f, -100, -ripple, where=(f>=wp1)&(f<=wp2), color='red', alpha=0.1)
    a.fill_between(f, -atenuacion, 5, where=(f<=ws1)|(f>=ws2), color='red', alpha=0.1)
    a.axhline(-ripple, color='gray', ls='--')
    a.axhline(-atenuacion, color='gray', ls='--')
    a.set_xlim(limites)
    a.set_ylim(-100, 5)
    a.set_xlabel('Frecuencia [Hz]')
    a.grid(True)
    a.legend()
ax[0].set_ylabel('Módulo [dB]')
ax[0].set_title('Comparación de filtros IIR')
ax[1].set_title('Zoom en bajas frecuencias')
plt.tight_layout()
plt.show()

#%% Zoom en banda de paso
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(f, mod_butt, label='Butterworth')
ax.plot(f, mod_cheby2, label='Chebyshev II')
ax.plot(f, mod_cauer, label='Cauer')
ax.fill_between(f, -100, -ripple, where=(f>=wp1)&(f<=wp2), color='red', alpha=0.1)
ax.axhline(-ripple, color='gray', ls='--')
ax.set_xlim(1, 42)
ax.set_ylim(-15, 2.5)
ax.set_title('Zoom en banda de paso')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Módulo [dB]')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

#%% Diseño FIR - Método de ventanas
# Definimos la respuesta deseada mediante puntos en frecuencia.
# Los límites se ajustan dentro de las bandas de transición para cumplir la plantilla.
#dejamos un margen respecto d elos limites obligatorios
freq_win = [0, 0.4, 0.9, 38, 47, nyq_frec]  # Frecuencias [Hz]
gain_win = [0, 0, 1, 1, 0, 0]  # Ganancia deseada en veces

# FIR con ventana Hamming
taps_win = 3501  # Cantidad de coeficientes
b_win = signal.firwin2(taps_win, freq_win, gain_win, window='hamming', fs=fs_ecg)

# FIR con ventana Blackman
taps_black = 5001
b_black = signal.firwin2(taps_black, freq_win, gain_win, window='blackman', fs=fs_ecg)

#Respuestas en frecuencia
_, H_win = signal.freqz(b_win, worN=W)
_, H_black = signal.freqz(b_black, worN=W)
mod_win = 20*np.log10(np.maximum(np.abs(H_win), 1e-12))
mod_black = 20*np.log10(np.maximum(np.abs(H_black), 1e-12))

# Verificación de la plantilla y demora
for nombre, taps, mod in [('Hamming', taps_win, mod_win), ('Blackman', taps_black, mod_black)]:
    print('\nFIR', nombre)
    print('Orden:', taps-1)
    print('Mínimo en paso:', round(np.min(mod[paso]), 2), 'dB')
    print('Máximo rechazo inferior:', round(np.max(mod[rechazo_inf]), 2), 'dB')
    print('Máximo rechazo superior:', round(np.max(mod[rechazo_sup]), 2), 'dB')
    print('Retardo de grupo:', (taps-1)/(2*fs_ecg), 's')


#%% Diseño FIR - Mínimos cuadrados
taps_firls = 2501  # Cantidad de coeficientes
# Incluimos las transiciones para evitar picos excesivos
freq_firls = [0, ws1, ws1, wp1, wp1, wp2, wp2, ws2, ws2, nyq_frec]
gain_firls = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
pesos = [100, 0.01, 1, 0.01, 10]  # Rechazo inf., transición, paso, transición, rechazo sup.

# Calculamos los coeficientes minimizando el error cuadrático ponderado
b_firls = signal.firls(taps_firls, freq_firls, gain_firls, weight=pesos, fs=fs_ecg)

# Normalizamos para que el máximo de la banda de paso sea 0 dB
_, H_temp = signal.freqz(b_firls, worN=W)
b_firls = b_firls/np.max(np.abs(H_temp[paso]))

# Respuesta en frecuencia
_, H_firls = signal.freqz(b_firls, worN=W)
mod_firls = 20*np.log10(np.maximum(np.abs(H_firls), 1e-12))

# Verificación de la plantilla
transicion_sup = (f>wp2)&(f<ws2)
print('\nFILTRO FIR - MÍNIMOS CUADRADOS')
print('Orden:', taps_firls-1)
print('Mínimo en paso:', round(np.min(mod_firls[paso]), 2), 'dB')
print('Máximo rechazo inferior:', round(np.max(mod_firls[rechazo_inf]), 2), 'dB')
print('Máximo rechazo superior:', round(np.max(mod_firls[rechazo_sup]), 2), 'dB')
print('Máximo en transición superior:', round(np.max(mod_firls[transicion_sup]), 2), 'dB')
print('Retardo de grupo:', (taps_firls-1)/(2*fs_ecg), 's')

#%% Diseño FIR - Parks-McClellan (Remez)
numtaps = 501 
ripple_relajado = 3         
atenuacion_relajada = 20    

ws1_rel = 0.5  
wp1_rel = 4.0  
wp2_rel = 35
ws2_rel = 40

bandas = [0, ws1_rel, wp1_rel, wp2_rel, ws2_rel, fs_ecg/2]
deseado = [0, 1, 0]
pesos = [1, 1, 1] 

b_remez_directo = signal.remez(
    numtaps, bandas, deseado, weight=pesos, fs=fs_ecg, 
    maxiter=500, grid_density=256
)
f_prueba = np.unique(np.r_[0, np.logspace(-3, np.log10(fs_ecg/2), 20000), ws1_rel, wp1_rel, wp2_rel, ws2_rel])

_, H_temp = signal.freqz(b_remez_directo, worN=f_prueba, fs=fs_ecg)
paso_temp = (f_prueba >= wp1_rel) & (f_prueba <= wp2_rel)
b_remez_directo = b_remez_directo / np.max(np.abs(H_temp[paso_temp]))

_, H_remez = signal.freqz(b_remez_directo, worN=f_prueba, fs=fs_ecg)
mod_remez = 20*np.log10(np.maximum(np.abs(H_remez), 1e-15))

paso_rel = (f_prueba>=wp1_rel)&(f_prueba<=wp2_rel)
rechazo_inf_rel = f_prueba<=ws1_rel
rechazo_sup_rel = f_prueba>=ws2_rel

print('\nFILTRO FIR - PARKS-MCCLELLAN')
print('Orden:', len(b_remez_directo)-1)
print('Ganancia mínima en paso:', round(np.min(mod_remez[paso_rel]), 2), 'dB')
print('Máximo rechazo inferior:', round(np.max(mod_remez[rechazo_inf_rel]), 2), 'dB')
print('Máximo rechazo superior:', round(np.max(mod_remez[rechazo_sup_rel]), 2), 'dB')
print('Retardo de grupo:', (len(b_remez_directo)-1)/(2*fs_ecg), 's')


#%% Comparación de los cuatro filtros FIR con ambas plantillas
from matplotlib.patches import Patch

filtros_fir = [
    ('Hamming', f, mod_win),
    ('Blackman', f, mod_black),
    ('Mínimos cuadrados', f, mod_firls),
    ('Parks-McClellan', f_prueba, mod_remez)
]

# Plantilla relajada: sombreado rojo claro (solo Parks-McClellan)
# Plantilla original: rayado diagonal rojo (restantes filtros)

def dibujar_plantillas_comparacion(a, ylim):
    ymin, ymax = ylim

    # PLANTILLA RELAJADA - sombreado rojo claro
    a.fill_between([0.001, ws1_rel], -atenuacion_relajada, ymax,
                   color='salmon', alpha=0.20, zorder=0)
    a.fill_between([wp1_rel, wp2_rel], ymin, -ripple_relajado,
                   color='salmon', alpha=0.20, zorder=0)
    a.fill_between([ws2_rel, nyq_frec], -atenuacion_relajada, ymax,
                   color='salmon', alpha=0.20, zorder=0)

    # PLANTILLA ORIGINAL - rayado diagonal rojo
    estilo_original = dict(facecolor='none', edgecolor='darkred',
                           hatch='////', linewidth=0, zorder=1)

    a.fill_between([0.001, ws1], -atenuacion, ymax, **estilo_original)
    a.fill_between([wp1, wp2], ymin, -ripple, **estilo_original)
    a.fill_between([ws2, nyq_frec], -atenuacion, ymax, **estilo_original)

# Etiquetas de las plantillas
leyenda_plantillas = [
    Patch(facecolor='none', edgecolor='darkred', hatch='////',
          label='Plantilla original (1 dB / 40 dB)'),
    Patch(facecolor='salmon', alpha=0.25,
          label='Plantilla relajada (3 dB / 20 dB) - Parks-McClellan')
]

# Gráficos comparativos
graficos = [
    ('Respuesta general', (0.01, 100), (-140, 15)),
    ('Bajas frecuencias', (0.01, 6), (-100, 5)),
    ('Banda de paso', (0.5, 35), (-5, 2))
]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))

for a, (titulo, xlim, ylim) in zip(ax, graficos):
    dibujar_plantillas_comparacion(a, ylim)

    for nombre, vector_f, mod in filtros_fir:
        a.plot(vector_f, mod, label=nombre, linewidth=1.2,
               color='tab:purple' if nombre=='Parks-McClellan' else None,
               zorder=3)

    if titulo != 'Banda de paso':
        a.set_xscale('log')

    a.set_xlim(xlim)
    a.set_ylim(ylim)
    a.set_title(titulo)
    a.set_xlabel('Frecuencia [Hz]')
    a.grid(True, which='both', ls=':', alpha=0.6)

ax[0].set_ylabel('Módulo [dB]')

# Leyenda conjunta de filtros y plantillas
curvas, etiquetas = ax[0].get_legend_handles_labels()
ax[0].legend(curvas + leyenda_plantillas,
             etiquetas + [h.get_label() for h in leyenda_plantillas],
             fontsize=7, loc='lower left')

plt.tight_layout()
plt.show()

#%% Comparación FIR vs IIR - Respuesta en frecuencia
from matplotlib.patches import Patch

# Si ya definiste esto arriba, no hace falta repetirlo.
# Lo dejo igual para que esta celda funcione sola.
def dibujar_plantillas_comparacion(a, ylim):
    ymin, ymax = ylim

    # Plantilla relajada (solo Parks-McClellan): sombreado rojo claro
    a.fill_between([0.001, ws1_rel], -atenuacion_relajada, ymax,
                   color='salmon', alpha=0.20, zorder=0)
    a.fill_between([wp1_rel, wp2_rel], ymin, -ripple_relajado,
                   color='salmon', alpha=0.20, zorder=0)
    a.fill_between([ws2_rel, nyq_frec], -atenuacion_relajada, ymax,
                   color='salmon', alpha=0.20, zorder=0)

    # Plantilla original: rayado diagonal rojo
    estilo_original = dict(facecolor='none', edgecolor='darkred',
                           hatch='////', linewidth=0, zorder=1)

    a.fill_between([0.001, ws1], -atenuacion, ymax, **estilo_original)
    a.fill_between([wp1, wp2], ymin, -ripple, **estilo_original)
    a.fill_between([ws2, nyq_frec], -atenuacion, ymax, **estilo_original)

leyenda_plantillas = [
    Patch(facecolor='none', edgecolor='darkred', hatch='////',
          label='Plantilla original (1 dB / 40 dB)'),
    Patch(facecolor='salmon', alpha=0.25,
          label='Plantilla relajada (3 dB / 20 dB) - Parks-McClellan')
]

# La comparación incluye cinco curvas por claridad; los siete filtros ya fueron diseñados.
filtros_comp = [
    ('Butterworth (IIR)', f, mod_butt),
    ('Cauer (IIR)', f, mod_cauer),
    ('Hamming (FIR)', f, mod_win),
    ('Mínimos cuadrados (FIR)', f, mod_firls),
    ('Parks-McClellan (FIR, relajado)', f_prueba, mod_remez)
]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
graficos = [
    ('Respuesta general', (0.01, 100), (-120, 10)),
    ('Bajas frecuencias', (0.01, 6), (-100, 5)),
    ('Banda de paso', (0.5, 35), (-5, 2))
]

for a, (titulo, xlim, ylim) in zip(ax, graficos):
    dibujar_plantillas_comparacion(a, ylim)

    for nombre, vector_f, modulo in filtros_comp:
        estilo = '--' if 'FIR' in nombre else '-'
        color = 'tab:purple' if 'Parks-McClellan' in nombre else None
        a.plot(vector_f, modulo, ls=estilo, label=nombre,
               linewidth=1.2, color=color, zorder=3)

    if titulo != 'Banda de paso':
        a.set_xscale('log')

    a.set_xlim(xlim)
    a.set_ylim(ylim)
    a.set_title(titulo)
    a.set_xlabel('Frecuencia [Hz]')
    a.grid(True, which='both', ls=':', alpha=0.6)

ax[0].set_ylabel('Módulo [dB]')

curvas, etiquetas = ax[0].get_legend_handles_labels()
ax[0].legend(curvas + leyenda_plantillas,
             etiquetas + [h.get_label() for h in leyenda_plantillas],
             fontsize=7, loc='lower left')

plt.tight_layout()
plt.show()


#%% Comparación FIR vs IIR - Fase y retardo de grupo
# Analizamos 4-35 Hz: banda de paso compartida por los siete diseños
f_paso = np.linspace(wp1_rel, wp2_rel, 6000)
W_paso = 2*np.pi*f_paso/fs_ecg  # Frecuencia digital [rad/muestra]

filtros_iir = [
    ('Butterworth', sos_butt),
    ('Chebyshev II', sos_cheby2),
    ('Cauer', sos_cauer)
]

filtros_fir = [
    ('Hamming', b_win),
    ('Blackman', b_black),
    ('Mínimos cuadrados', b_firls),
    ('Parks-McClellan', b_remez_directo)
]

fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex='col')

# Filtros IIR: fase no lineal y retardo variable
for nombre, sos in filtros_iir:
    _, H = signal.freqz_sos(sos, worN=W_paso)
    fase = np.unwrap(np.angle(H))
    demora = -np.gradient(fase, W_paso)*1000/fs_ecg  # Retardo [ms]

    ax[0,0].plot(f_paso, fase-fase[0], label=nombre)
    ax[1,0].plot(f_paso, demora, label=nombre)

    print(nombre, '- Retardo mínimo:', round(np.min(demora), 2),
          'ms - Máximo:', round(np.max(demora), 2), 'ms')

# Filtros FIR: fase lineal y retardo constante
for nombre, b in filtros_fir:
    _, H = signal.freqz(b, worN=W_paso)
    fase = np.unwrap(np.angle(H))
    demora = (len(b)-1)/(2*fs_ecg)*1000  # Retardo [ms]

    ax[0,1].plot(f_paso, fase-fase[0], label=nombre)
    ax[1,1].plot(f_paso, np.full_like(f_paso, demora), label=nombre)

    print(nombre, '- Retardo:', round(demora, 2), 'ms')

# Formato de los gráficos
for a in ax.flat:
    a.grid(True, ls=':')
    a.legend(fontsize=8)
    a.set_xlim(wp1_rel, wp2_rel)

ax[0,0].set_title('Fase - IIR')
ax[0,1].set_title('Fase - FIR')
ax[1,0].set_title('Retardo de grupo - IIR')
ax[1,1].set_title('Retardo de grupo - FIR')

ax[0,0].set_ylabel('Fase relativa [rad]')
ax[0,1].set_ylabel('Fase relativa [rad]')
ax[1,0].set_ylabel('Retardo [ms]')
ax[1,1].set_ylabel('Retardo [ms]')

ax[1,0].set_xlabel('Frecuencia [Hz]')
ax[1,1].set_xlabel('Frecuencia [Hz]')
ax[1,0].set_yscale('log')

plt.tight_layout()
plt.show()
#%% Carga del ECG
datos = sio.loadmat('ECG_TP4.mat')
ecg_one_lead = datos['ecg_lead'].flatten().astype(float)
cant_muestras = len(ecg_one_lead)

print('Cantidad de muestras:', cant_muestras)
print('Duración del registro:', cant_muestras/fs_ecg/60, 'min')


#%% Aplicación de los filtros al ECG

# Filtros IIR: una pasada causal con condiciones iniciales
filtros_iir = [('Butterworth', sos_butt), ('Chebyshev II', sos_cheby2), ('Cauer', sos_cauer)]
ecg_iir = {}
for nombre, sos in filtros_iir:
    zi = signal.sosfilt_zi(sos)*ecg_one_lead[0]
    ecg_iir[nombre], _ = signal.sosfilt(sos, ecg_one_lead, zi=zi)

# Filtros FIR: convolución causal y compensación de demora
# CLAVE: Actualizamos a b_remez_directo
filtros_fir = [
    ('Hamming', b_win), 
    ('Blackman', b_black), 
    ('Mínimos cuadrados', b_firls), 
    ('Parks-McClellan', b_remez_directo) 
]
ecg_fir = {}
for nombre, b in filtros_fir:
    filtrada = signal.oaconvolve(ecg_one_lead, b, mode='full')[:cant_muestras]
    demora = (len(b)-1)//2  # Demora de grupo [muestras]
    alineada = np.full(cant_muestras, np.nan)
    alineada[:cant_muestras-demora] = filtrada[demora:]
    ecg_fir[nombre] = alineada


#%% Regiones de interés con ruido
regs_interes = ([4000, 5500], [10000, 11000])

for inicio, fin in regs_interes:
    zoom = np.arange(inicio, min(fin, cant_muestras))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Comparación ECG original vs IIR
    ax[0].plot(zoom, ecg_one_lead[zoom], color='gray', alpha=0.65, linewidth=2, label='ECG original')
    for nombre, filtrada in ecg_iir.items():
        ax[0].plot(zoom, filtrada[zoom], label=nombre, linewidth=1.2)
    ax[0].set_title('Filtrado IIR - Región con ruido')
    ax[0].set_ylabel('Amplitud')

    # Comparación ECG original vs FIR
    ax[1].plot(zoom, ecg_one_lead[zoom], color='gray', alpha=0.65, linewidth=2, label='ECG original')
    for nombre, filtrada in ecg_fir.items():
        ax[1].plot(zoom, filtrada[zoom], label=nombre, linewidth=1.2)
    ax[1].set_title('Filtrado FIR - Región con ruido (demora compensada)')
    ax[1].set_xlabel('Muestras')
    ax[1].set_ylabel('Amplitud')

    for a in ax:
        a.grid(True, ls=":")
        a.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Región de interés: muestras {inicio} a {fin}')
    plt.tight_layout()
    plt.show()
    #%% Filtrado IIR con fase cero - sosfiltfilt
# Como trabajamos fuera de tiempo real, usamos doble pasada para eliminar el desfase
# Diseñamos con la mitad de las atenuaciones, ya que sosfiltfilt duplica los dB

ecg_iir_cero = {}
sos_iir_cero = {}

for nombre, tipo in [('Butterworth', 'butter'), ('Chebyshev II', 'cheby2'), ('Cauer', 'ellip')]:
    sos = signal.iirdesign(wp, ws, gpass=ripple/2, gstop=atenuacion/2, ftype=tipo, fs=fs_ecg, output='sos')
    sos_iir_cero[nombre] = sos
    ecg_iir_cero[nombre] = signal.sosfiltfilt(sos, ecg_one_lead)
#%% Comparación IIR - Filtrado causal vs fase cero
regs_interes = ([4000, 5500], [10000, 11000])

for inicio, fin in regs_interes:
    zoom = np.arange(inicio, min(fin, cant_muestras))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)

    # IIR de una pasada: conservan el retardo y la distorsión de fase
    ax[0].plot(zoom, ecg_one_lead[zoom], color='gray', lw=2, alpha=0.6, label='ECG original')
    for nombre, filtrada in ecg_iir.items():
        ax[0].plot(zoom, filtrada[zoom], lw=1.2, label=nombre)
    ax[0].set_title('IIR causal - sosfilt')

    # IIR de doble pasada: fase cero
    ax[1].plot(zoom, ecg_one_lead[zoom], color='gray', lw=2, alpha=0.6, label='ECG original')
    for nombre, filtrada in ecg_iir_cero.items():
        ax[1].plot(zoom, filtrada[zoom], lw=1.2, label=nombre)
    ax[1].plot(zoom, ecg_fir['Hamming'][zoom], 'k--', lw=1.2, label='FIR Hamming (referencia)')
    ax[1].set_title('IIR fase cero - sosfiltfilt')
    ax[1].set_xlabel('Muestras')

    for a in ax:
        a.set_ylabel('Amplitud')
        a.grid(True)
        a.legend(fontsize=8)
    fig.suptitle(f'Región de interés: muestras {inicio} a {fin}')
    plt.tight_layout()
    plt.show()

#%% Verificación de los IIR con fase cero
f_ver = np.unique(np.r_[np.linspace(0, 3, 6001), np.linspace(3, 100, 10000), np.linspace(100, nyq_frec, 10000), ws1, wp1, wp2, ws2])
W_ver = 2*np.pi*f_ver/fs_ecg

for nombre, sos in sos_iir_cero.items():
    _, H = signal.freqz_sos(sos, worN=W_ver)
    modulo = 40*np.log10(np.maximum(np.abs(H), 1e-12))  # Módulo de doble pasada [dB]
    paso_ver = (f_ver>=wp1)&(f_ver<=wp2)
    rechazo_inf_ver = f_ver<=ws1
    rechazo_sup_ver = f_ver>=ws2
    print('\nIIR FASE CERO -', nombre)
    print('Mínimo en paso:', round(np.min(modulo[paso_ver]), 2), 'dB')
    print('Máximo en paso:', round(np.max(modulo[paso_ver]), 2), 'dB')
    print('Máximo rechazo inferior:', round(np.max(modulo[rechazo_inf_ver]), 2), 'dB')
    print('Máximo rechazo superior:', round(np.max(modulo[rechazo_sup_ver]), 2), 'dB')


#%% Regiones de interés sin ruido
regs_limpias = (
    np.array([5, 5.2])*60*fs_ecg,
    np.array([12, 12.4])*60*fs_ecg,
    np.array([15, 15.2])*60*fs_ecg
)

for ii in regs_limpias:
    inicio = max(0, int(ii[0]))
    fin = min(cant_muestras, int(ii[1]))
    if inicio>=fin:
        print('Región fuera del registro:', ii)
        continue
    zoom = np.arange(inicio, fin)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ECG original vs IIR con fase cero
    ax[0].plot(zoom, ecg_one_lead[zoom], color='gray', lw=2, alpha=0.6, label='ECG original')
    for nombre, filtrada in ecg_iir_cero.items():
        ax[0].plot(zoom, filtrada[zoom], lw=1.2, label=nombre)
    ax[0].set_title('Filtrado IIR - Fase cero')

    # ECG original vs FIR con demora compensada
    ax[1].plot(zoom, ecg_one_lead[zoom], color='gray', lw=2, alpha=0.6, label='ECG original')
    for nombre, filtrada in ecg_fir.items():
        ax[1].plot(zoom, filtrada[zoom], lw=1.2, label=nombre)
    ax[1].set_title('Filtrado FIR - Demora compensada')
    ax[1].set_xlabel('Muestras')

    for a in ax:
        a.set_ylabel('Amplitud')
        a.grid(True)
        a.legend(fontsize=8)
    fig.suptitle(f'Región sin ruido: muestras {inicio} a {fin}')
    plt.tight_layout()
    plt.show()


#%% Zoom para analizar la morfología del ECG
inicio, fin = 309900, 310950
zoom = np.arange(inicio, fin)

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)

# IIR con fase cero
ax[0].plot(zoom, ecg_one_lead[zoom], color='gray', lw=2, alpha=0.7, label='ECG original')
for nombre, filtrada in ecg_iir_cero.items():
    ax[0].plot(zoom, filtrada[zoom], label=nombre)
ax[0].set_title('IIR con fase cero - Morfología del ECG')

# FIR con demora compensada
ax[1].plot(zoom, ecg_one_lead[zoom], color='gray', lw=2, alpha=0.7, label='ECG original')
for nombre, filtrada in ecg_fir.items():
    ax[1].plot(zoom, filtrada[zoom], label=nombre)
ax[1].set_title('FIR con demora compensada - Morfología del ECG')
ax[1].set_xlabel('Muestras')

for a in ax:
    a.set_ylabel('Amplitud')
    a.grid(True)
    a.legend(fontsize=8)
plt.tight_layout()
plt.show()
#%% Tabla comparativa final - Usamos los filtros ya diseñados, sin rediseñarlos
import pandas as pd
from IPython.display import display

def medidas_tabla(frec, respuesta_db, fp_inf, fp_sup, fr_inf, fr_sup,
                  max_paso, min_rechazo):
    banda_paso = (frec>=fp_inf)&(frec<=fp_sup)
    banda_inf = frec<=fr_inf
    banda_sup = frec>=fr_sup
    minimo_paso = np.min(respuesta_db[banda_paso])
    maximo_paso = np.max(respuesta_db[banda_paso])
    rechazo_inferior = -np.max(respuesta_db[banda_inf])
    rechazo_superior = -np.max(respuesta_db[banda_sup])
    cumple = (minimo_paso >= -max_paso-0.01 and maximo_paso <= 0.01 and
              rechazo_inferior >= min_rechazo-0.01 and
              rechazo_superior >= min_rechazo-0.01)
    return round(maximo_paso-minimo_paso, 2), round(rechazo_inferior, 2), round(rechazo_superior, 2), cumple

filas = []
# IIR: orden inicial causal y orden del rediseño aplicado con sosfiltfilt.
for nombre, sos_causal in [('Butterworth', sos_butt), ('Chebyshev II', sos_cheby2), ('Cauer', sos_cauer)]:
    sos = sos_iir_cero[nombre]
    _, H = signal.freqz_sos(sos, worN=W_ver)
    mod = 40*np.log10(np.maximum(np.abs(H), 1e-12))  # respuesta efectiva de doble pasada
    variacion, rech_inf, rech_sup, cumple = medidas_tabla(
        f_ver, mod, wp1, wp2, ws1, ws2, ripple, atenuacion)
    filas.append([nombre + ' (IIR)', 'Original', 2*len(sos_causal), 2*len(sos),
                  '0 neto (doble pasada)', variacion, rech_inf, rech_sup,
                  'Sí' if cumple else 'No', 'Sí' if cumple else 'No'])

# FIR de una pasada con la plantilla original.
for nombre, b in [('Hamming', b_win), ('Blackman', b_black), ('Mínimos cuadrados', b_firls)]:
    _, H = signal.freqz(b, worN=W)
    mod = 20*np.log10(np.maximum(np.abs(H), 1e-12))
    variacion, rech_inf, rech_sup, cumple = medidas_tabla(
        f, mod, wp1, wp2, ws1, ws2, ripple, atenuacion)
    orden = len(b)-1
    filas.append([nombre + ' (FIR)', 'Original', orden, orden,
                  f'{orden/(2*fs_ecg)*1000:.0f} ms', variacion, rech_inf, rech_sup,
                  'Sí' if cumple else 'No', 'Sí' if cumple else 'No'])

# Parks-McClellan: sólo cambian sus propias especificaciones, no las de los otros filtros.
_, H = signal.freqz(b_remez_directo, worN=f_prueba, fs=fs_ecg)
mod = 20*np.log10(np.maximum(np.abs(H), 1e-15))
variacion, rech_inf, rech_sup, cumple_rel = medidas_tabla(
    f_prueba, mod, wp1_rel, wp2_rel, ws1_rel, ws2_rel,
    ripple_relajado, atenuacion_relajada)
_, H_orig = signal.freqz(b_remez_directo, worN=f, fs=fs_ecg)
mod_orig = 20*np.log10(np.maximum(np.abs(H_orig), 1e-15))
_, _, _, cumple_orig = medidas_tabla(
    f, mod_orig, wp1, wp2, ws1, ws2, ripple, atenuacion)
orden = len(b_remez_directo)-1
filas.append(['Parks-McClellan (FIR)', 'Relajada', orden, orden,
              f'{orden/(2*fs_ecg)*1000:.0f} ms', variacion, rech_inf, rech_sup,
              'Sí' if cumple_rel else 'No', 'Sí' if cumple_orig else 'No'])

columnas = ['Filtro', 'Plantilla', 'Orden causal/inicial', 'Orden aplicado',
            'Retardo de aplicación', 'Variación en paso [dB]',
            'Rechazo inferior [dB]', 'Rechazo superior [dB]',
            'Cumple propia plantilla', 'Cumple original']
df_comparacion = pd.DataFrame(filas, columns=columnas)
print('Tabla comparativa de los filtros diseñados:')
display(df_comparacion.style.hide(axis='index'))



