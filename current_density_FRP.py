import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

# Updated to point to the new RFP data folder
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

print("Loading RFP data...")
fields = readsav(input_folder + 'modeprofiles.sav')

# UPDATED KEYS for RFP data
t = fields['t'].view()
br = np.transpose(fields['br'].view(), (2, 1, 0))
bt = np.transpose(fields['bt'].view(), (2, 1, 0))
bz = np.transpose(fields['bz'].view(), (2, 1, 0))
r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

mode = janz(0, 0, input_folder)

# Use the final time step to see the fully formed RFP state
t_idx = -1
current_time = t[t_idx]

# ==========================================
# 1. Calculating Current Densities
# Assuming mu_0 = 1, so J = curl(B)
# ==========================================

# Extract the 1D arrays for the m=0, n=0 mode at the mature time step
bz_00 = bz[t_idx, :, mode]
bt_00 = bt[t_idx, :, mode]

# Calculate J_theta = - d(B_z)/dr
j_theta = -np.gradient(bz_00, r2)

# Calculate J_z = (B_theta / r) + d(B_theta)/dr
dbt_dr = np.gradient(bt_00, r2)
b_theta_over_r = np.zeros(r2.shape, dtype=complex)

tolerance = 1e-10
safe_mask = r2 > tolerance
np.divide(bt_00, r2, out=b_theta_over_r, where=safe_mask)

# Apply L'Hopital's rule for the origin limit
zero_mask = r2 <= tolerance
b_theta_over_r[zero_mask] = dbt_dr[zero_mask]

# J_z is the sum of the two terms
j_z = b_theta_over_r + dbt_dr

# Plot the Current Densities
plt.figure(figsize=(8, 6))
plt.plot(r2, j_z.real, color='purple', label=r"$J_z$", linestyle='-', linewidth=2)
plt.plot(r2, j_theta.real, color='orange', label=r"$J_{\theta}$", linestyle='-', linewidth=2)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.xlabel('Radius (r)')
plt.ylabel('Current Density (J)')
plt.title(f'RFP Equilibrium Current Profiles (m=0, n=0)\nTime = {current_time:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.show()

# ==========================================
# 2. Parallel and Perpendicular Components
# ==========================================

bz_real = bz_00.real
bt_real = bt_00.real
jz_real = j_z.real
jt_real = j_theta.real

B_sq = bz_real**2 + bt_real**2
J_dot_B = jz_real * bz_real + jt_real * bt_real

alpha = np.zeros_like(B_sq)
np.divide(J_dot_B, B_sq, out=alpha, where=(B_sq != 0))

j_par_t = alpha * bt_real
j_par_z = alpha * bz_real

j_perp_t = jt_real - j_par_t
j_perp_z = jz_real - j_par_z

# Mask out the r=0 point for plotting
mask = (r2 != 0)
r2_plot = r2[mask]

jz_plot = jz_real[mask]
j_par_z_plot = j_par_z[mask]
j_perp_z_plot = j_perp_z[mask]

jt_plot = jt_real[mask]
j_par_t_plot = j_par_t[mask]
j_perp_t_plot = j_perp_t[mask]

# Plotting the separated components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Z components
ax1.plot(r2_plot, jz_plot, color='purple', label=r"$J_z$ (Total)", linestyle='-', linewidth=2)
ax1.plot(r2_plot, j_par_z_plot, color='blue', label=r"$J_{\parallel, z}$", linestyle='--')
ax1.plot(r2_plot, j_perp_z_plot, color='red', label=r"$J_{\perp, z}$", linestyle='-.')
ax1.axhline(0, color='black', linewidth=1, linestyle=':')
ax1.set_xlabel('Radius (r)')
ax1.set_ylabel('Current Density')
ax1.set_title(f'RFP Axial Current Density Components\nTime = {current_time:.2f}')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Subplot 2: Theta components
ax2.plot(r2_plot, jt_plot, color='orange', label=r"$J_{\theta}$ (Total)", linestyle='-', linewidth=2)
ax2.plot(r2_plot, j_par_t_plot, color='blue', label=r"$J_{\parallel, \theta}$", linestyle='--')
ax2.plot(r2_plot, j_perp_t_plot, color='red', label=r"$J_{\perp, \theta}$", linestyle='-.')
ax2.axhline(0, color='black', linewidth=1, linestyle=':')
ax2.set_xlabel('Radius (r)')
ax2.set_ylabel('Current Density')
ax2.set_title(f'RFP Poloidal Current Density Components\nTime = {current_time:.2f}')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.show()