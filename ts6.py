import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# =============================================================================
# Parámetros del circuito a)
# =============================================================================
w0 = 1.0
Q = np.sqrt(2)/2

# H(s) = 1 / (s^2 + sqrt(2)s + 1)
num = [1]
den = [1, np.sqrt(2), 1]

sys_a = signal.TransferFunction(num, den)

# =============================================================================
# Respuesta en frecuencia
# =============================================================================

# Solo frecuencias positivas
w = np.logspace(-2, 2, 1000)

w_a, mag_a, phase_a = signal.bode(sys_a, w)

# Respuesta asintótica del módulo:
# 0 dB hasta w0 y -40 dB/dec para w > w0
asintota_mag = np.where(
    w <= w0,
    0,
    -40*np.log10(w/w0)
)

# Nivel de corte exacto
nivel_corte = 20*np.log10(1/np.sqrt(2))

# Polos y ceros
polos = sys_a.poles
ceros = sys_a.zeros

# =============================================================================
# Gráficos
# =============================================================================

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# -----------------------------------------------------------------------------
# 1. Módulo
# -----------------------------------------------------------------------------

ax[0].semilogx(
    w_a,
    mag_a,
    linewidth=2,
    label='Respuesta de módulo'
)

ax[0].semilogx(
    w,
    asintota_mag,
    '--',
    linewidth=1.5,
    label='Respuesta asintótica del módulo'
)

# Frecuencia de corte
ax[0].axvline(
    w0,
    color='tab:red',
    linestyle=':',
    label=r'$\omega_c=\omega_0=1$ rad/s'
)

# Nivel de -3.01 dB
ax[0].axhline(
    nivel_corte,
    color='tab:green',
    linestyle=':',
    label=r'Nivel de corte: $-3.01$ dB'
)

# Punto de corte
ax[0].plot(
    w0,
    nivel_corte,
    'o',
    color='tab:green'
)

# Pendiente asintótica
ax[0].annotate(
    r'$-40$ dB/dec',
    xy=(10, -40),
    xytext=(10, 8),
    textcoords='offset points',
    fontsize=10
)

ax[0].set_title('Módulo - Circuito a) Pasa bajos')
ax[0].set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax[0].set_ylabel(r'$|H(j\omega)|$ [dB]')
ax[0].grid(True, which='both', linestyle='--', alpha=0.7)
ax[0].legend(loc='lower left')


# -----------------------------------------------------------------------------
# 2. Fase
# -----------------------------------------------------------------------------

ax[1].semilogx(
    w_a,
    phase_a,
    linewidth=2,
    label='Respuesta de fase'
)

ax[1].axvline(
    w0,
    color='tab:red',
    linestyle=':',
    label=r'$\omega_0=1$ rad/s'
)

ax[1].axhline(
    -90,
    color='tab:green',
    linestyle=':',
    label=r'$\phi(\omega_0)=-90^\circ$'
)

ax[1].plot(
    w0,
    -90,
    'o',
    color='tab:green'
)

ax[1].set_title('Fase - Circuito a) Pasa bajos')
ax[1].set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax[1].set_ylabel(r'Fase $\phi(\omega)$ [$^\circ$]')
ax[1].grid(True, which='both', linestyle='--', alpha=0.7)
ax[1].legend(loc='lower left')


# -----------------------------------------------------------------------------
# 3. Polos y ceros
# -----------------------------------------------------------------------------

theta = np.linspace(0, 2*np.pi, 200)

# Circunferencia de radio w0
ax[2].plot(
    w0*np.cos(theta),
    w0*np.sin(theta),
    ':',
    label=r'$|s|=\omega_0=1$'
)

# Ejes del plano complejo
ax[2].axhline(0, color='black', linewidth=1)
ax[2].axvline(0, color='black', linewidth=1)

# Polos
ax[2].plot(
    np.real(polos),
    np.imag(polos),
    'x',
    markersize=10,
    markeredgewidth=2,
    label='Polos'
)

# Ceros
if len(ceros) > 0:
    ax[2].plot(
        np.real(ceros),
        np.imag(ceros),
        'o',
        markersize=8,
        markerfacecolor='none',
        label='Ceros'
    )

ax[2].set_title('Diagrama de polos y ceros')
ax[2].set_xlabel('Parte real')
ax[2].set_ylabel('Parte imaginaria')
ax[2].grid(True, linestyle='--', alpha=0.7)
ax[2].axis('equal')
ax[2].legend(loc='upper right')

plt.tight_layout()
plt.show()

#%% CIRCUITO B

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# =============================================================================
# 1. Parámetros del circuito b)
# =============================================================================
w0 = 1.0
Q = np.sqrt(2)/2
wz = np.sqrt(2)

# Frecuencia de corte
wc = np.sqrt(-2 + np.sqrt(5))

# H(s) = (s^2 + sqrt(2)s) / (s^2 + sqrt(2)s + 1)
num = [1, np.sqrt(2), 0]
den = [1, np.sqrt(2), 1]

sys_b = signal.TransferFunction(num, den)

# =============================================================================
# 2. Respuesta en frecuencia
# =============================================================================
w = np.logspace(-2, 2, 1000)
w_b, mag_b, phase_b = signal.bode(sys_b, w)

# Nivel de corte
nivel_corte = 20*np.log10(1/np.sqrt(2))

# Valor exacto del módulo en w0
mag_w0 = 20*np.log10(np.sqrt(3/2))

# Valor exacto de la fase en w0
fase_w0 = 35.26

# =============================================================================
# 3. Respuesta asintótica del módulo
# =============================================================================

asintota_mag = np.zeros_like(w)

# Para w < w0:
# H ≈ sqrt(2)*s  --> +20 dB/dec
zona1 = w < w0
asintota_mag[zona1] = 20*np.log10(np.sqrt(2)*w[zona1])

# Para w0 <= w < sqrt(2):
# los dos polos agregan -40 dB/dec
# pendiente neta = +20 - 40 = -20 dB/dec
zona2 = (w >= w0) & (w < wz)
asintota_mag[zona2] = (
    20*np.log10(np.sqrt(2))
    - 20*np.log10(w[zona2]/w0)
)

# Para w >= sqrt(2):
# +20 +20 -40 = 0 dB/dec
zona3 = w >= wz
asintota_mag[zona3] = 0

# =============================================================================
# 4. Polos y ceros
# =============================================================================
polos = sys_b.poles
ceros = sys_b.zeros

# =============================================================================
# 5. Gráficos
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# -----------------------------------------------------------------------------
# MÓDULO
# -----------------------------------------------------------------------------
ax[0].semilogx(
    w_b,
    mag_b,
    linewidth=2,
    label='Respuesta de módulo'
)

# Asíntota para altas frecuencias
ax[0].axhline(
    0,
    linestyle='--',
    color='black',
    alpha=0.8,
    label='Asíntota: 0 dB'
)

# Nivel de corte
ax[0].axhline(
    nivel_corte,
    linestyle=':',
    color='tab:green',
    label=r'Nivel de corte: $-3.01$ dB'
)

# Frecuencia de corte
ax[0].axvline(
    wc,
    linestyle=':',
    color='tab:green',
    label=rf'$\omega_c={wc:.3f}$ rad/s'
)

# Frecuencia natural
ax[0].axvline(
    w0,
    linestyle=':',
    color='tab:red',
    label=r'$\omega_0=1$ rad/s'
)

# Punto exacto en w0
ax[0].plot(
    w0,
    mag_w0,
    'o',
    color='tab:red'
)

ax[0].annotate(
    r'$|H(j\omega_0)|=1.76$ dB',
    xy=(w0, mag_w0),
    xytext=(10, 10),
    textcoords='offset points'
)

# Pendiente para bajas frecuencias
ax[0].annotate(
    r'$+20$ dB/dec',
    xy=(0.05, -22),
    xytext=(0, 0),
    textcoords='offset points',
    fontsize=10,
    fontweight='bold',
    rotation=28
)

ax[0].set_title('Módulo - Circuito b)')
ax[0].set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax[0].set_ylabel(r'$|H(j\omega)|$ [dB]')
ax[0].grid(True, which='both', linestyle='--', alpha=0.7)
ax[0].legend(loc='lower right', fontsize=8)

# -----------------------------------------------------------------------------
# FASE
# -----------------------------------------------------------------------------
ax[1].semilogx(
    w_b,
    phase_b,
    linewidth=2,
    label='Respuesta de fase'
)

# Frecuencia natural
ax[1].axvline(
    w0,
    linestyle=':',
    color='tab:red',
    label=r'$\omega_0=1$ rad/s'
)

# Fase exacta en w0
ax[1].axhline(
    fase_w0,
    linestyle=':',
    color='tab:green',
    label=rf'$\phi(\omega_0)={fase_w0:.1f}^\circ$'
)

ax[1].plot(
    w0,
    fase_w0,
    'o',
    color='tab:green'
)

# Referencias asintóticas
ax[1].axhline(
    90,
    linestyle='--',
    color='gray',
    alpha=0.7,
    label=r'Bajas frecuencias: $90^\circ$'
)

ax[1].axhline(
    0,
    linestyle='--',
    color='black',
    alpha=0.7,
    label=r'Altas frecuencias: $0^\circ$'
)

ax[1].set_title('Fase - Circuito b)')
ax[1].set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax[1].set_ylabel(r'Fase $\phi(\omega)$ [$^\circ$]')
ax[1].grid(True, which='both', linestyle='--', alpha=0.7)
ax[1].legend(loc='upper right', fontsize=8)


# -----------------------------------------------------------------------------
# POLOS Y CEROS
# -----------------------------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 200)

# Circunferencia de radio w0
ax[2].plot(
    w0*np.cos(theta),
    w0*np.sin(theta),
    ':',
    label=r'$|s|=\omega_0=1$'
)

# Ejes del plano complejo
ax[2].axhline(0, color='black', linewidth=1)
ax[2].axvline(0, color='black', linewidth=1)

# Polos
ax[2].plot(
    np.real(polos),
    np.imag(polos),
    'rx',
    markersize=10,
    markeredgewidth=2,
    label='Polos'
)

# Ceros
ax[2].plot(
    np.real(ceros),
    np.imag(ceros),
    'bo',
    markersize=8,
    markerfacecolor='none',
    markeredgewidth=2,
    label='Ceros'
)

ax[2].set_title('Diagrama de polos y ceros')
ax[2].set_xlabel(r'Parte real de $s$')
ax[2].set_ylabel(r'Parte imaginaria de $s$')

ax[2].set_xlim(-1.7, 1.2)
ax[2].set_ylim(-1.2, 1.2)

ax[2].grid(True, linestyle='--', alpha=0.7)
ax[2].axis('equal')
ax[2].legend(loc='upper right')

plt.tight_layout()
plt.show()

#%% CIRCUITO C

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# =============================================================================
# 1. Parámetros del circuito c)
# =============================================================================
w0 = 1.0
Q = np.sqrt(2)/2

# Frecuencias de corte
w1 = np.sqrt(2 - np.sqrt(3))
w2 = np.sqrt(2 + np.sqrt(3))

# Función de transferencia:
# H(s) = (sqrt(2)s) / (s^2 + sqrt(2)s + 1)
num = [np.sqrt(2), 0]
den = [1, np.sqrt(2), 1]

sys_c = signal.TransferFunction(num, den)

# =============================================================================
# 2. Respuesta en frecuencia
# =============================================================================
w = np.logspace(-2, 2, 1000)
w_c, mag_c, phase_c = signal.bode(sys_c, w)

# Polos y ceros
polos = sys_c.poles
ceros = sys_c.zeros

# Nivel de corte
nivel_corte = -3.01

# =============================================================================
# 3. Asíntota del módulo
# =============================================================================
asintota_mag = np.zeros_like(w)

# Para w < w0: +20 dB/dec
zona1 = w <= w0
asintota_mag[zona1] = 20*np.log10(w[zona1] / w0)

# Para w > w0: -20 dB/dec
zona2 = w > w0
asintota_mag[zona2] = -20*np.log10(w[zona2] / w0)

# =============================================================================
# 4. Gráficos
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# -----------------------------------------------------------------------------
# MÓDULO
# -----------------------------------------------------------------------------
ax[0].semilogx(
    w_c,
    mag_c,
    linewidth=2,
    label='Respuesta de módulo'
)

ax[0].semilogx(
    w,
    asintota_mag,
    '--',
    linewidth=1.5,
    label='Respuesta asintótica del módulo'
)

# Frecuencia natural
ax[0].axvline(
    w0,
    linestyle=':',
    color='tab:red',
    label=r'$\omega_0=1$ rad/s'
)

# Frecuencias de corte
ax[0].axvline(
    w1,
    linestyle=':',
    color='tab:green',
    label=rf'$\omega_1={w1:.3f}$ rad/s'
)

ax[0].axvline(
    w2,
    linestyle=':',
    color='tab:purple',
    label=rf'$\omega_2={w2:.3f}$ rad/s'
)

# Nivel de corte
ax[0].axhline(
    nivel_corte,
    linestyle=':',
    color='black',
    label=r'Nivel de corte: $-3.01$ dB'
)

# Punto en w0
ax[0].plot(
    w0,
    0,
    'o',
    color='tab:red'
)

ax[0].annotate(
    r'Resonancia: $0$ dB',
    xy=(w0, 0),
    xytext=(10, -10),
    textcoords='offset points'
)

# Pendientes
ax[0].annotate(
    r'$+20$ dB/dec',
    xy=(0.05, -22),
    xytext=(0, 0),
    textcoords='offset points',
    fontsize=10,
    fontweight='bold',
    rotation=33
)

ax[0].annotate(
    r'$-20$ dB/dec',
    xy=(10, -12),
    xytext=(0, 0),
    textcoords='offset points',
    fontsize=10,
    fontweight='bold',
    rotation=-33
)

ax[0].set_title('Módulo - Circuito c) Pasa banda')
ax[0].set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax[0].set_ylabel(r'$|H(j\omega)|$ [dB]')
ax[0].grid(True, which='both', linestyle='--', alpha=0.7)
ax[0].legend(loc='lower left', fontsize=8)


# -----------------------------------------------------------------------------
# FASE
# -----------------------------------------------------------------------------
ax[1].semilogx(
    w_c,
    phase_c,
    linewidth=2,
    label='Respuesta de fase'
)

# Frecuencia natural
ax[1].axvline(
    w0,
    linestyle=':',
    color='tab:red',
    label=r'$\omega_0=1$ rad/s'
)

# Referencias
ax[1].axhline(
    90,
    linestyle='--',
    color='gray',
    alpha=0.8,
    label=r'Bajas frecuencias: $90^\circ$'
)

ax[1].axhline(
    0,
    linestyle=':',
    color='tab:green',
    label=r'$\phi(\omega_0)=0^\circ$'
)

ax[1].axhline(
    -90,
    linestyle='--',
    color='black',
    alpha=0.8,
    label=r'Altas frecuencias: $-90^\circ$'
)

# Punto en w0
ax[1].plot(
    w0,
    0,
    'o',
    color='tab:green'
)

ax[1].annotate(
    r'$0^\circ$',
    xy=(w0, 0),
    xytext=(8, 8),
    textcoords='offset points'
)

ax[1].set_title('Fase - Circuito c) Pasa banda')
ax[1].set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax[1].set_ylabel(r'Fase $\phi(\omega)$ [$^\circ$]')
ax[1].grid(True, which='both', linestyle='--', alpha=0.7)
ax[1].legend(loc='upper right', fontsize=8)


# -----------------------------------------------------------------------------
# POLOS Y CEROS
# -----------------------------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 200)

# Circunferencia de radio w0
ax[2].plot(
    w0*np.cos(theta),
    w0*np.sin(theta),
    ':',
    label=r'$|s|=\omega_0=1$'
)

# Ejes
ax[2].axhline(0, color='black', linewidth=1)
ax[2].axvline(0, color='black', linewidth=1)

# Polos
ax[2].plot(
    np.real(polos),
    np.imag(polos),
    'rx',
    markersize=10,
    markeredgewidth=2,
    label='Polos'
)

# Ceros
if len(ceros) > 0:
    ax[2].plot(
        np.real(ceros),
        np.imag(ceros),
        'bo',
        markersize=8,
        markerfacecolor='none',
        markeredgewidth=2,
        label='Ceros'
    )

ax[2].set_title('Diagrama de polos y ceros')
ax[2].set_xlabel(r'Parte real de $s$')
ax[2].set_ylabel(r'Parte imaginaria de $s$')
ax[2].set_xlim(-1.2, 1.2)
ax[2].set_ylim(-1.1, 1.1)
ax[2].grid(True, linestyle='--', alpha=0.7)
ax[2].axis('equal')
ax[2].legend(loc='upper right')

plt.tight_layout()
plt.show()
#%% CIRCUITO BONUS A=B=1

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# =============================================================================
# 1. Parámetros - Filtro elimina banda / Notch
# =============================================================================
w0 = 1.0
Q = np.sqrt(2) / 2

# Para a = b = 1:
# H(s) = (s^2 + 1) / (s^2 + sqrt(2)s + 1)

num = [1, 0, 1]
den = [1, np.sqrt(2), 1]

sys_notch = signal.TransferFunction(num, den)

# Frecuencias de corte calculadas analíticamente
w1 = np.sqrt(2 - np.sqrt(3))
w2 = np.sqrt(2 + np.sqrt(3))

# =============================================================================
# 2. Respuesta en frecuencia
# =============================================================================

# Solo frecuencias positivas
w = np.logspace(-2, 2, 2000)

# Evaluación directa de H(jw)
s = 1j*w
H = (s**2 + 1)/(s**2 + np.sqrt(2)*s + 1)

# Módulo en dB
mag = 20*np.log10(np.abs(H))

# Fase
phase = np.angle(H, deg=True)

# Asíntota de módulo:
# el filtro tiende a 0 dB tanto para bajas como para altas frecuencias
asintota_mag = np.zeros_like(w)

# Nivel de corte
nivel_corte = 20*np.log10(1/np.sqrt(2))

# =============================================================================
# 3. Gráficos
# =============================================================================
fig = plt.figure(figsize=(15, 5))

# -----------------------------------------------------------------------------
# PANEL 1: Módulo
# -----------------------------------------------------------------------------
ax1 = plt.subplot(1, 3, 1)

ax1.semilogx(w, mag, 'b-', linewidth=2,
             label='Respuesta real')

ax1.semilogx(w, asintota_mag, 'k--', linewidth=1.5,
             alpha=0.8, label='Asíntota: 0 dB')

# Nivel de -3 dB
ax1.axhline(nivel_corte, color='r', linestyle=':',
            label=r'Nivel de corte: $-3.01$ dB')

# Frecuencias de corte
ax1.axvline(w1, color='g', linestyle=':',
            label=rf'$\omega_1={w1:.3f}$ rad/s')

ax1.axvline(w2, color='orange', linestyle=':',
            label=rf'$\omega_2={w2:.3f}$ rad/s')

# Frecuencia del notch
ax1.axvline(w0, color='purple', linestyle=':',
            label=r'Notch: $\omega_0=1$ rad/s')

# Puntos de -3 dB
ax1.plot(w1, nivel_corte, 'ro')
ax1.plot(w2, nivel_corte, 'ro')

# Indicar rechazo total
ax1.annotate(r'$|H(j\omega_0)|=0$' '\n' r'$(-\infty$ dB)',
             xy=(w0, -55),
             xytext=(8, 5),
             textcoords='offset points',
             color='purple',
             fontsize=10,
             fontweight='bold')

ax1.set_title('1. Módulo - Filtro Notch')
ax1.set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax1.set_ylabel('Magnitud [dB]')
ax1.set_ylim(-60, 5)
ax1.grid(True, which='both', linestyle='--', alpha=0.7)
ax1.legend(loc='lower left')


# -----------------------------------------------------------------------------
# PANEL 2: Fase
# -----------------------------------------------------------------------------
ax2 = plt.subplot(1, 3, 2)

ax2.semilogx(w, phase, 'b-', linewidth=2,
             label='Respuesta de fase')

ax2.axvline(w0, color='purple', linestyle=':',
            label=r'$\omega_0=1$ rad/s')

ax2.axhline(0, color='gray', linestyle='--',
            label=r'Fase $=0^\circ$')

ax2.set_title('2. Fase - Filtro Notch')
ax2.set_xlabel(r'Frecuencia angular $\omega$ [rad/s]')
ax2.set_ylabel(r'Fase [$^\circ$]')
ax2.grid(True, which='both', linestyle='--', alpha=0.7)
ax2.legend(loc='best')


# -----------------------------------------------------------------------------
# PANEL 3: Polos y Ceros
# -----------------------------------------------------------------------------
ax3 = plt.subplot(1, 3, 3)

polos = sys_notch.poles
ceros = sys_notch.zeros

# Circunferencia de radio w0
theta = np.linspace(0, 2*np.pi, 200)

ax3.plot(w0*np.cos(theta),
         w0*np.sin(theta),
         'k:',
         label=r'$|s|=\omega_0=1$')

# Ejes cartesianos
ax3.axhline(0, color='black', linewidth=1)
ax3.axvline(0, color='black', linewidth=1)

# Polos
ax3.plot(np.real(polos),
         np.imag(polos),
         'rx',
         markersize=10,
         markeredgewidth=2,
         label='Polos')

# Ceros
ax3.plot(np.real(ceros),
         np.imag(ceros),
         'bo',
         markersize=9,
         markerfacecolor='none',
         markeredgewidth=2,
         label='Ceros')

ax3.set_title('3. Diagrama de Polos y Ceros')
ax3.set_xlabel(r'Parte real de $s$')
ax3.set_ylabel(r'Parte imaginaria de $s$')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.axis('equal')
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()