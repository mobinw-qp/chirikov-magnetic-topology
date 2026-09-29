from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt

input_folder = "C:/data.specly/"

def janz(m, n, directory):
    """Return the value of the array corresponding to the specified mode (m,n)"""
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
# Load Data
# ==========================================
fields = readsav(input_folder + 'modeprofiles_10000.sav')
t = fields['prot'].view()

# Arrays are transposed to shape: (time, radius, mode)
br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))

r2 = fields['pror2'].view()
mode = janz(0, 0, input_folder)

# ==========================================
# STEP 1: Identify your Sawtooth Indices
# ==========================================
# This plots the time evolution of the core Bz (at r=0).
# Hover over the plot to find the exact array indices (x-axis) for your crash.

plt.figure(num="Figure 1: Core Bz Evolution", figsize=(10, 4))
plt.plot(range(len(t)), bz[:, 0, mode].real, label="Bz at r=0")
plt.title("Core Bz Evolution")
plt.xlabel("Array Index (Time Step)")
plt.ylabel("Bz (m=0, n=0)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show(block=False) # block=False allows the script to continue to the next plots

# ==========================================
# STEP 2: Define the indices for your cycle
# ==========================================
# *** UPDATE THESE NUMBERS *** based on what you see in Figure 1.
idx_start = 550   # Start of the linear tearing phase
idx_peak = 690    # Right before the crash
idx_end = 740     # Right after the crash

time_indices = [idx_start, idx_peak, idx_end]
labels = ['Linear Phase', 'Pre-Crash', 'Post-Crash']
colors = ['blue', 'orange', 'red'] 
line_styles = [':', '--', '-']

# ==========================================
# STEP 3: Loop through times and plot profiles
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, num="Figure 2: Current Profiles", figsize=(14, 6))
tolerance = 1e-10

for i, t_idx in enumerate(time_indices):
    # 1. Extract 1D arrays for this specific time step
    bz_00 = bz[t_idx, :, mode]
    bt_00 = bt[t_idx, :, mode]
    
    # 2. Calculate Currents (J = curl B)
    j_theta = -np.gradient(bz_00, r2)
    dbt_dr = np.gradient(bt_00, r2)
    
    b_theta_over_r = np.zeros(r2.shape, dtype=complex)
    
    # Safely handle the division by r
    safe_mask = r2 > tolerance
    np.divide(bt_00, r2, out=b_theta_over_r, where=safe_mask)
    
    # Apply L'Hopital's rule at the origin
    zero_mask = r2 <= tolerance
    b_theta_over_r[zero_mask] = dbt_dr[zero_mask]
    
    j_z = b_theta_over_r + dbt_dr
    
    # 3. Real parts for physical calculations
    bz_real = bz_00.real
    bt_real = bt_00.real
    jz_real = j_z.real
    jt_real = j_theta.real
    
    # 4. Parallel / Perpendicular Math
    B_sq = bz_real**2 + bt_real**2
    J_dot_B = jz_real * bz_real + jt_real * bt_real
    
    alpha = np.zeros_like(B_sq)
    np.divide(J_dot_B, B_sq, out=alpha, where=(B_sq != 0))
    
    j_par_t = alpha * bt_real
    j_par_z = alpha * bz_real
    j_perp_t = jt_real - j_par_t
    j_perp_z = jz_real - j_par_z

    # 5. Plotting parallel components over time
    # Notice we plot 'r2' directly now because the origin is mathematically fixed!
    ax1.plot(r2, j_par_z, color=colors[i], linestyle=line_styles[i], 
             label=f"{labels[i]} (t={t[t_idx]:.2f})")
    
    ax2.plot(r2, j_par_t, color=colors[i], linestyle=line_styles[i], 
             label=f"{labels[i]} (t={t[t_idx]:.2f})")

# Formatting Subplot 1 (Z)
ax1.axhline(0, color='black', linewidth=1, linestyle='-')
ax1.set_xlabel('Radius (r)')
ax1.set_ylabel(r'$J_{\parallel, z}$')
ax1.set_title('Evolution of Parallel Z Current Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Formatting Subplot 2 (Theta)
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_xlabel('Radius (r)')
ax2.set_ylabel(r'$J_{\parallel, \theta}$')
ax2.set_title('Evolution of Parallel Theta Current Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# ==========================================
# NEW CODE: Safety Factor (q) Evolution
# ==========================================

# Define your machine's major radius / aspect ratio (R_0 / a)
# Update this value to match your simulation settings!
R = 4.0 

plt.figure(figsize=(8, 6))

for i, t_idx in enumerate(time_indices):
    # 1. Extract 1D arrays for this specific time step (real parts)
    bz_00 = bz[t_idx, :, mode].real
    bt_00 = bt[t_idx, :, mode].real
    
    # 2. Calculate the derivative of B_theta for the q(0) boundary condition
    dbt_dr = np.gradient(bt_00, r2)
    
    # 3. Initialize the q array
    q_profile = np.zeros_like(r2)
    
    # 4. Safe calculation for r > 0
    safe_mask = r2 > tolerance
    q_profile[safe_mask] = (r2[safe_mask] * bz_00[safe_mask]) / (R * bt_00[safe_mask])
    
    # 5. Apply L'Hopital's rule for r = 0
    zero_mask = r2 <= tolerance
    q_profile[zero_mask] = bz_00[zero_mask] / (R * dbt_dr[zero_mask])
    
    # 6. Plot the profile
    plt.plot(r2, q_profile, color=colors[i], linestyle=line_styles[i], 
             label=f"{labels[i]} (t={t[t_idx]:.2f})")

# Formatting the Plot
# Adding a horizontal line at q=1, which is often the rational surface where sawteeth crash
plt.axhline(1.0, color='gray', linewidth=1, linestyle='-', alpha=0.5, label='q=1 surface')
plt.axhline(0.0, color='black', linewidth=1, linestyle='-')

plt.xlabel('Radius (r)')
plt.ylabel('Safety Factor (q)')
plt.title(f'Evolution of Safety Factor q-profile (R={R})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()