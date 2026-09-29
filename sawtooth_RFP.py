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
# Load Data (Using updated RFP keys)
# ==========================================
fields = readsav(input_folder + 'modeprofiles.sav')
t = fields['t'].view()

br = np.transpose(fields['br'].view(), (2, 1, 0))
bt = np.transpose(fields['bt'].view(), (2, 1, 0))
bz = np.transpose(fields['bz'].view(), (2, 1, 0))

r2 = fields['pror2'].view()
mode = janz(0, 0, input_folder)

# ==========================================
# STEP 1: Identify your RFP Evolution Indices
# ==========================================
# Instead of a sawtooth crash, an RFP dynamically relaxes into its reversed state.
# We will capture the beginning, middle, and very end (t=3000) of this process.

idx_start = 0                 # Startup phase
idx_peak = len(t) // 2        # Mid-point of the dynamo relaxation
idx_end = -1                  # Fully mature RFP state

time_indices = [idx_start, idx_peak, idx_end]
labels = ['Startup Phase', 'Relaxation Phase', 'Mature RFP']
colors = ['blue', 'orange', 'red'] 
line_styles = [':', '--', '-']

# Plot the evolution of the core Bz to show how the equilibrium settles
plt.figure(num="Figure 1: Core Bz Evolution", figsize=(10, 4))
plt.plot(t, bz[:, 0, mode].real, label="Bz at r=0 (Core)")
plt.plot(t, bz[:, -1, mode].real, label="Bz at r=a (Edge)", color='purple', linestyle='--')
plt.axhline(0, color='black', linestyle='-', alpha=0.5)
plt.title("RFP Axial Magnetic Field Evolution")
plt.xlabel("Time (t)")
plt.ylabel("Bz (m=0, n=0)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show(block=False) 

# ==========================================
# STEP 2: Loop through times and plot current profiles
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, num="Figure 2: Current Profiles", figsize=(14, 6))
tolerance = 1e-10

for i, t_idx in enumerate(time_indices):
    bz_00 = bz[t_idx, :, mode]
    bt_00 = bt[t_idx, :, mode]
    
    # Calculate Currents (J = curl B)
    j_theta = -np.gradient(bz_00, r2)
    dbt_dr = np.gradient(bt_00, r2)
    
    b_theta_over_r = np.zeros(r2.shape, dtype=complex)
    
    safe_mask = r2 > tolerance
    np.divide(bt_00, r2, out=b_theta_over_r, where=safe_mask)
    
    zero_mask = r2 <= tolerance
    b_theta_over_r[zero_mask] = dbt_dr[zero_mask]
    
    j_z = b_theta_over_r + dbt_dr
    
    # Real parts for physical calculations
    bz_real = bz_00.real
    bt_real = bt_00.real
    jz_real = j_z.real
    jt_real = j_theta.real
    
    # Parallel / Perpendicular Math
    B_sq = bz_real**2 + bt_real**2
    J_dot_B = jz_real * bz_real + jt_real * bt_real
    
    alpha = np.zeros_like(B_sq)
    np.divide(J_dot_B, B_sq, out=alpha, where=(B_sq != 0))
    
    j_par_t = alpha * bt_real
    j_par_z = alpha * bz_real

    # Plotting parallel components over time
    ax1.plot(r2, j_par_z, color=colors[i], linestyle=line_styles[i], 
             label=f"{labels[i]} (t={t[t_idx]:.0f})")
    
    ax2.plot(r2, j_par_t, color=colors[i], linestyle=line_styles[i], 
             label=f"{labels[i]} (t={t[t_idx]:.0f})")

ax1.axhline(0, color='black', linewidth=1, linestyle='-')
ax1.set_xlabel('Radius (r)')
ax1.set_ylabel(r'$J_{\parallel, z}$')
ax1.set_title('Evolution of Parallel Z Current Density')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_xlabel('Radius (r)')
ax2.set_ylabel(r'$J_{\parallel, \theta}$')
ax2.set_title('Evolution of Parallel Theta Current Density')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.show(block=False)

# ==========================================
# STEP 3: Safety Factor (q) Evolution
# ==========================================
R = 4.0 

plt.figure(num="Figure 3: RFP q-profile", figsize=(8, 6))

for i, t_idx in enumerate(time_indices):
    bz_00 = bz[t_idx, :, mode].real
    bt_00 = bt[t_idx, :, mode].real
    
    dbt_dr = np.gradient(bt_00, r2)
    q_profile = np.zeros_like(r2)
    
    safe_mask = r2 > tolerance
    q_profile[safe_mask] = (r2[safe_mask] * bz_00[safe_mask]) / (R * bt_00[safe_mask])
    
    zero_mask = r2 <= tolerance
    q_profile[zero_mask] = bz_00[zero_mask] / (R * dbt_dr[zero_mask])
    
    plt.plot(r2, q_profile, color=colors[i], linestyle=line_styles[i], 
             label=f"{labels[i]} (t={t[t_idx]:.0f})")

# In an RFP, q is always less than 1. The critical surface is where q=0 (the reversal surface).
plt.axhline(0.0, color='black', linewidth=1.5, linestyle='-', label='q=0 (Reversal Surface)')

plt.xlabel('Radius (r)')
plt.ylabel('Safety Factor (q)')
plt.title(f'Evolution of RFP Safety Factor q-profile (R={R})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.show()