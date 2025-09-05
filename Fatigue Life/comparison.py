import numpy as np
import matplotlib.pyplot as plt
from strain_life import strain_life_diagram

# -------------------------------
# 1) Energy-based method
# -------------------------------
E = 210e3
sigma_y = 470
sigma_u = 745
epsilon_f = 0.22

epsilon_true_f = np.log(1 + epsilon_f)
epsilon_py = 0.002
epsilon_pu = epsilon_true_f - sigma_u / E
n_energy = np.log(epsilon_pu / epsilon_py) / np.log(sigma_u / sigma_y)
K_energy = sigma_y / (epsilon_py ** (1 / n_energy))

Wf = (sigma_u * epsilon_true_f
      - sigma_u**2 / (2 * E)
      - 0.002 * sigma_u**((n_energy) + 1) / (((n_energy) + 1) * sigma_y**(n_energy)))

epsilon_cal = 0.012
def strain_from_stress(sigma):
    return sigma / E + (sigma / K_energy) ** n_energy

def sigma_from_strain_scan(epsilon_target):
    sigma_values = np.linspace(1, 2000, 20000)
    strain_values = [strain_from_stress(s) for s in sigma_values]
    idx = np.argmin(np.abs(np.array(strain_values) - epsilon_target))
    return sigma_values[idx]

sigma_cal = sigma_from_strain_scan(epsilon_cal)
Wh_cal = (sigma_cal * epsilon_cal
          - sigma_cal**2 / (2 * E)
          - 0.002 * sigma_cal**((n_energy) + 1) / (((n_energy) + 1) * sigma_y**(n_energy)))
K_life = 1030 * Wh_cal / Wf

stress_levels = np.arange(0, 900, 50)
strain_amplitudes_energy = []
lives_energy = []
for sigma_c in stress_levels:
    epsilon_c = strain_from_stress(sigma_c)
    Wh = (sigma_c * epsilon_c
          - sigma_c**2 / (2 * E)
          - 0.002 * sigma_c**((n_energy) + 1) / (((n_energy) + 1) * sigma_y**(n_energy)))
    Nf = K_life * Wf / Wh
    strain_amplitudes_energy.append(epsilon_c)
    lives_energy.append(Nf)

# -------------------------------
# 2) Coffin–Manson–Basquin method
# -------------------------------
E_cmb = 210000
sigma_f = 745
epsilon_f_cmb = 0.563
b = -0.095
c = -0.563

# Build object but suppress its internal plot
sld = strain_life_diagram(
    E=E_cmb,
    sigma_f=sigma_f,
    epsilon_f=epsilon_f_cmb,
    b=b,
    c=c,
    show_plot=False
)

# Recompute the same arrays the class uses
cycles_2Nf_array = np.logspace(1, 8, 1000)
strain_amplitude_cmb = (sigma_f / E_cmb) * cycles_2Nf_array**b + epsilon_f_cmb * cycles_2Nf_array**c
lives_cmb = cycles_2Nf_array / 2

# -------------------------------
# Combined Plot
# -------------------------------
plt.figure(figsize=(8,6))

plt.loglog(lives_energy, strain_amplitudes_energy, 'o-', label="Energy Method (4340)")
plt.loglog(lives_cmb, strain_amplitude_cmb, 'r-', label="Coffin-Manson-Basquin")

plt.xlabel("Fatigue Life, Nf (cycles)")
plt.ylabel("Strain Amplitude (εa)")
plt.title("Comparison of Strain-Life Curves")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
