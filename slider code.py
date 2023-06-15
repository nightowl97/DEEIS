import matplotlib.pyplot as plt
import numpy as np
from main import *
from matplotlib.widgets import Slider
matplotlib.use("TkAgg")

cpe_freq, = read_csv('CPE.csv')

# Generate synthetic data
freq = np.logspace(-3, 5, 100)
w = 2 * np.pi * freq

fig = plt.figure(figsize=(4, 6))
ax = fig.add_axes([0, 0.40, 1, 0.6])

rhf = 0.6
rlf = 4.2
tau = 550
alpha = .5
nu = .5
beta = .5
q = 1e-5


def get_z(rhf, rlf, tau, alpha, nu, beta, q):
    # Cole-Davidson
    # z = rhf + (rlf - rhf) / (1 + (1j * tau * w) ** alpha)
    # Havriliak-Negami
    z = rhf + (rlf - rhf) / ((1 + (1j * w * tau) ** nu) ** beta) + (1 / (q * (1j * w) ** alpha))
    return z


z = get_z(rhf, rlf, tau, alpha, nu, beta, q)
line = ax.scatter(z.real, -z.imag)
ax.grid()

rhf_slider_ax = fig.add_axes([0.2, 0.01, 0.55, 0.06])
rlf_slider_ax = fig.add_axes([0.2, 0.05, 0.55, 0.06])
tau_slider_ax = fig.add_axes([0.2, 0.1, 0.55, 0.06])
alpha_slider_ax = fig.add_axes([0.2, 0.15, 0.55, 0.06])
nu_slider_ax = fig.add_axes([0.2, 0.2, 0.55, 0.06])
beta_slider_ax = fig.add_axes([0.2, 0.25, 0.55, 0.06])
q_slider_ax = fig.add_axes([0.2, 0.3, 0.55, 0.06])

rhf_slider = Slider(ax=rhf_slider_ax, label='Rhf', valmin=0, valmax=5, valstep=.1, valinit=rhf)
rlf_slider = Slider(ax=rlf_slider_ax, label='Rlf', valmin=0, valmax=5, valstep=.1, valinit=rlf)
tau_slider = Slider(ax=tau_slider_ax, label='tau', valmin=50, valmax=600, valstep=1, valinit=tau)
alpha_slider = Slider(ax=alpha_slider_ax, label='alpha', valmin=0, valmax=1, valstep=.01, valinit=alpha)
nu_slider = Slider(ax=nu_slider_ax, label='nu', valmin=0, valmax=1, valstep=.01, valinit=nu)
beta_slider = Slider(ax=beta_slider_ax, label='beta', valmin=0, valmax=1, valstep=.01, valinit=beta)
q_slider = Slider(ax=q_slider_ax, label='q', valmin=1e-6, valmax=1e-4, valstep=1e-5, valinit=q)


def sliders_on_changed(val):
    z = get_z(rhf_slider.val, rlf_slider.val, tau_slider.val, alpha_slider.val, nu_slider.val, beta_slider.val,
              q_slider.val)
    line.set_offsets(np.c_[z.real, -z.imag])
    fig.canvas.draw_idle()


rhf_slider.on_changed(sliders_on_changed)
rlf_slider.on_changed(sliders_on_changed)
tau_slider.on_changed(sliders_on_changed)
alpha_slider.on_changed(sliders_on_changed)
nu_slider.on_changed(sliders_on_changed)
beta_slider.on_changed(sliders_on_changed)
q_slider.on_changed(sliders_on_changed)


plt.show()