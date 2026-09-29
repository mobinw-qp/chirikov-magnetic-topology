import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

input_folder = "C:/data.specly/RFP/"

def janz(m, n, directory):
    """Return the value of the array corresponding to the specified mode (m,n)"""
    if not directory.endswith('/'):
        directory += '/'
    spectr_arrays = readsav(directory + 'spectrum.sav')
    my = spectr_arrays['MY'].view()
    mm = spectr_arrays['MM'].view()
    nanf = spectr_arrays['NANF'].view()
    nz = spectr_arrays['NZ'].view()

    j = 0
    if m == 0 and n == 0:
        return int(j)
    for im in range(m):
        j = j + nz[im]
    
    j = j + n - nanf[m] + 1
    return int(j)

# ==========================================
# Load Data (Using updated RFP dictionary keys)
# ==========================================
print("Loading RFP data...")
fields = readsav(input_folder + 'modeprofiles.sav')
t = fields['t'].view()

br = np.transpose(fields['br'].view(), (2, 1, 0))
bt = np.transpose(fields['bt'].view(), (2, 1, 0))
bz = np.transpose(fields['bz'].view(), (2, 1, 0))
vr = np.transpose(fields['vr'].view(), (2, 1, 0))
vt = np.transpose(fields['vt'].view(), (2, 1, 0))
vz = np.transpose(fields['vz'].view(), (2, 1, 0))

r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

mode_00 = janz(0, 0, input_folder)

# In an RFP, the dominant tearing modes are usually m=1 with negative n (e.g., n=-6, -7, -8).
# You can change n_val below to match the exact dominant instability of your simulation.
m_val, n_val = 1, -6
mode_dominant = janz(m_val, n_val, input_folder) 

# ==========================================
# 1. Plot Equilibrium Magnetic Fields
# ==========================================
# Using the final time step (-1) to show the mature RFP state
plt.figure(figsize=(8, 5))
plt.plot(r2, bz[-1, :, mode_00].real, color='blue', label=r"$B_z$", linestyle='-', linewidth=2)
plt.plot(r2, bt[-1, :, mode_00].real, color='red', label=r"$B_{\theta}$", linestyle='-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title(f"RFP Equilibrium Magnetic Fields (m=0, n=0)\nTime = {t[-1]:.2f}")
plt.xlabel("Radius (r)")
plt.ylabel("Magnetic Field Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.show(block=False)

# ==========================================
# 2. NEW CODE: Energy Calculations (Fixed for Staggered Grids)
# ==========================================
print("Calculating Macroscopic Energies...")

# Add axes to r1 and r2 so they broadcast correctly across (time, radius, mode)
r1_3d = r1[np.newaxis, :, np.newaxis]
r2_3d = r2[np.newaxis, :, np.newaxis]

# Calculate integrands separately based on their grids
# B_r and V_r use the r1 grid
integrand_M_r = 0.5 * np.abs(br)**2 * r1_3d
integrand_K_r = 0.5 * np.abs(vr)**2 * r1_3d

# B_theta, B_z, V_theta, and V_z use the r2 grid
integrand_M_tz = 0.5 * (np.abs(bt)**2 + np.abs(bz)**2) * r2_3d
integrand_K_tz = 0.5 * (np.abs(vt)**2 + np.abs(vz)**2) * r2_3d

# Integrate along their respective radial axes (axis=1) using trapz
energy_M_r_modes = np.trapz(integrand_M_r, x=r1, axis=1)
energy_M_tz_modes = np.trapz(integrand_M_tz, x=r2, axis=1)

energy_K_r_modes = np.trapz(integrand_K_r, x=r1, axis=1)
energy_K_tz_modes = np.trapz(integrand_K_tz, x=r2, axis=1)

# Sum the integrations to get the total mode energies (shape: time, mode)
energy_M_modes = energy_M_r_modes + energy_M_tz_modes
energy_K_modes = energy_K_r_modes + energy_K_tz_modes

# Calculate total macroscopic energies by summing across the mode axis (axis=1)
total_energy_M = np.sum(energy_M_modes, axis=1)
total_energy_K = np.sum(energy_K_modes, axis=1)

# ==========================================
# 3. Plotting the Energies
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Subplot 1: Magnetic Energy ---
ax1.plot(t, total_energy_M, label='Total Magnetic Energy', color='black', linewidth=2)
ax1.plot(t, energy_M_modes[:, mode_00], label='m=0, n=0 (Equilibrium)', color='purple')
ax1.plot(t, energy_M_modes[:, mode_dominant], label=f'm={m_val}, n={n_val} (Tearing Mode)', color='red')

ax1.set_yscale('log')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Magnetic Energy (Log Scale)')
ax1.set_title('RFP Evolution of Magnetic Energy')
ax1.grid(True, alpha=0.3, which="both")
ax1.legend()

# --- Subplot 2: Kinetic Energy ---
ax2.plot(t, total_energy_K, label='Total Kinetic Energy', color='black', linewidth=2)
ax2.plot(t, energy_K_modes[:, mode_00], label='m=0, n=0 (Equilibrium)', color='purple')
ax2.plot(t, energy_K_modes[:, mode_dominant], label=f'm={m_val}, n={n_val} (Tearing Mode)', color='red')

ax2.set_yscale('log')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Kinetic Energy (Log Scale)')
ax2.set_title('RFP Evolution of Kinetic Energy')
ax2.grid(True, alpha=0.3, which="both")
ax2.legend()

plt.tight_layout()
plt.show()