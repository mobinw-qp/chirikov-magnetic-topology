from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt

input_folder="C:/data.specly/"

def janz(m,n,directory):
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


fields = readsav(input_folder + 'modeprofiles_10000.sav')
t = fields['prot'].view()
br = np.transpose(fields['prodbr'].view(),(2,1,0))
bt = np.transpose(fields['prodbt'].view(),(2,1,0))
bz = np.transpose(fields['prodbz'].view(),(2,1,0))
vr = np.transpose(fields['prodvr'].view(),(2,1,0))
vt = np.transpose(fields['prodvt'].view(),(2,1,0))
vz = np.transpose(fields['prodvz'].view(),(2,1,0))

r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

mode = janz(0,0,input_folder)


plt.plot(r2, bz[0,:,mode], color='blue', label= r"$B_z$", linestyle='-')
plt.plot(r2, bt[0,:,mode], color='red', label= r"$B_{\theta}$", linestyle='-')
plt.legend()
plt.show()
# ==========================================
# NEW CODE: Energy Calculations (Fixed for Staggered Grids)
# ==========================================

# 1. Get the indices for the modes of interest
mode_00 = janz(0, 0, input_folder)
mode_11 = janz(1, 1, input_folder) # The primary tearing mode

# 2. Add axes to r1 and r2 so they broadcast correctly across (time, radius, mode)
r1_3d = r1[np.newaxis, :, np.newaxis]
r2_3d = r2[np.newaxis, :, np.newaxis]

# 3. Calculate integrands separately based on their grids
# B_r and V_r use the r1 grid
integrand_M_r = 0.5 * np.abs(br)**2 * r1_3d
integrand_K_r = 0.5 * np.abs(vr)**2 * r1_3d

# B_theta, B_z, V_theta, and V_z use the r2 grid
integrand_M_tz = 0.5 * (np.abs(bt)**2 + np.abs(bz)**2) * r2_3d
integrand_K_tz = 0.5 * (np.abs(vt)**2 + np.abs(vz)**2) * r2_3d

# 4. Integrate along their respective radial axes (axis=1) using trapz
energy_M_r_modes = np.trapz(integrand_M_r, x=r1, axis=1)
energy_M_tz_modes = np.trapz(integrand_M_tz, x=r2, axis=1)

energy_K_r_modes = np.trapz(integrand_K_r, x=r1, axis=1)
energy_K_tz_modes = np.trapz(integrand_K_tz, x=r2, axis=1)

# 5. Sum the integrations to get the total mode energies (shape: time, mode)
energy_M_modes = energy_M_r_modes + energy_M_tz_modes
energy_K_modes = energy_K_r_modes + energy_K_tz_modes

# 6. Calculate total macroscopic energies by summing across the mode axis (axis=1)
total_energy_M = np.sum(energy_M_modes, axis=1)
total_energy_K = np.sum(energy_K_modes, axis=1)

# ==========================================
# Plotting the Energies
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Subplot 1: Magnetic Energy ---
ax1.plot(t, total_energy_M, label='Total Magnetic Energy', color='black', linewidth=2)
ax1.plot(t, energy_M_modes[:, mode_00], label='m=0, n=0 (Equilibrium)', color='purple')
ax1.plot(t, energy_M_modes[:, mode_11], label='m=1, n=1 (Tearing Mode)', color='red')

ax1.set_yscale('log')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Magnetic Energy (Log Scale)')
ax1.set_title('Evolution of Magnetic Energy')
ax1.grid(True, alpha=0.3, which="both")
ax1.legend()

# --- Subplot 2: Kinetic Energy ---
ax2.plot(t, total_energy_K, label='Total Kinetic Energy', color='black', linewidth=2)
ax2.plot(t, energy_K_modes[:, mode_00], label='m=0, n=0 (Equilibrium)', color='purple')
ax2.plot(t, energy_K_modes[:, mode_11], label='m=1, n=1 (Tearing Mode)', color='red')

ax2.set_yscale('log')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Kinetic Energy (Log Scale)')
ax2.set_title('Evolution of Kinetic Energy')
ax2.grid(True, alpha=0.3, which="both")
ax2.legend()

plt.tight_layout()
plt.show()
# ==========================================
# NEW CODE: 2D Temporal Evolution (Spatiotemporal Heatmap)
# ==========================================

# 1. Extract the full (time, radius) arrays for the equilibrium mode
bz_00_all = bz[:, :, mode].real
bt_00_all = bt[:, :, mode].real

# 2. Vectorized calculation for the derivative of B_theta across the radial axis (axis=1)
dbt_dr_all = np.gradient(bt_00_all, r2, axis=1)

# 3. Initialize the 2D q-profile array (Time x Radius)
q_all = np.zeros_like(bz_00_all)
R = 4.0  # Make sure this matches the Major Radius from your previous code

# 4. Safe calculation for all r > 0 across all time steps
tolerance = 1e-10
safe_mask = r2 > tolerance
# NumPy broadcasting allows us to apply the 1D r2 mask to the 2D arrays
q_all[:, safe_mask] = (r2[safe_mask] * bz_00_all[:, safe_mask]) / (R * bt_00_all[:, safe_mask])

# 5. Apply L'Hopital's rule for the core (r = 0) across all time steps
zero_mask = r2 <= tolerance
q_all[:, zero_mask] = bz_00_all[:, zero_mask] / (R * dbt_dr_all[:, zero_mask])

# ==========================================
# Plotting the 2D Heatmap
# ==========================================
plt.figure(figsize=(12, 6))

# Create a meshgrid for the axes
T_grid, R_grid = np.meshgrid(t, r2)

# Plot the heatmap (transpose q_all so the axes match T and R)
heatmap = plt.pcolormesh(T_grid, R_grid, q_all.T, cmap='viridis', shading='nearest')

# Overlay a bold white line exactly where q = 1
contour = plt.contour(T_grid, R_grid, q_all.T, levels=[1.0], colors='white', linewidths=2)
plt.clabel(contour, inline=True, fontsize=12, fmt="q=1 Surface")

# Formatting
cbar = plt.colorbar(heatmap)
cbar.set_label('Safety Factor (q)', rotation=270, labelpad=15)

plt.xlabel('Time (t)')
plt.ylabel('Radius (r)')
plt.title('Spatiotemporal Evolution of the Safety Factor')

# Optional: Zoom in on a specific time window to see the sawteeth clearly
# plt.xlim(2000, 10000)

plt.show()
# ==========================================
# NEW CODE: 2D Temporal Evolution of J_z
# ==========================================

# 1. Extract the full (time, radius) array for B_theta (m=0, n=0)
bt_00_all = bt[:, :, mode].real

# 2. Vectorized calculation for the derivative of B_theta across the radial axis (axis=1)
dbt_dr_all = np.gradient(bt_00_all, r2, axis=1)

# 3. Initialize the division array (B_theta / r)
b_theta_over_r_all = np.zeros_like(bt_00_all)

# 4. Safe calculation for all r > 0 across all time steps
tolerance = 1e-10
safe_mask = r2 > tolerance
b_theta_over_r_all[:, safe_mask] = bt_00_all[:, safe_mask] / r2[safe_mask]

# 5. Apply L'Hopital's rule for the core (r = 0) across all time steps
zero_mask = r2 <= tolerance
b_theta_over_r_all[:, zero_mask] = dbt_dr_all[:, zero_mask]

# 6. Calculate the full J_z array (Time x Radius)
jz_all = b_theta_over_r_all + dbt_dr_all

# ==========================================
# Plotting the J_z 2D Heatmap
# ==========================================
plt.figure(figsize=(12, 6))

# Create a meshgrid for the axes (using the same t and r2 arrays)
T_grid, R_grid = np.meshgrid(t, r2)

# Plot the heatmap (transpose jz_all so the axes match T and R)
# 'plasma' is a great colormap for highlighting intense core currents
heatmap_jz = plt.pcolormesh(T_grid, R_grid, jz_all.T, cmap='plasma', shading='nearest')

# Formatting
cbar_jz = plt.colorbar(heatmap_jz)
cbar_jz.set_label(r'Axial Current Density ($J_z$)', rotation=270, labelpad=20)

plt.xlabel('Time (t)')
plt.ylabel('Radius (r)')
plt.title('Spatiotemporal Evolution of Axial Current Density ($J_z$)')

plt.show()
# 1. Define your physical grid in ordinary space
theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta)

# FIX: Define the specific time index you want to reconstruct
t_idx = 350  # Replace 350 with the exact index of your crash

# 2. Extract both real and imaginary parts of your modes at that specific time
bz_00_real = bz[t_idx, :, mode_00].real

# Need both real and imag for modes where m > 0 or n > 0
bz_11_real = bz[t_idx, :, mode_11].real
bz_11_imag = bz[t_idx, :, mode_11].imag

# 3. Initialize the empty physical field array
Bz_physical = np.zeros_like(R_grid)

# ... (rest of the code continues below)
# 1. Define your physical grid in ordinary space
theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta)

# 2. Extract both real and imaginary parts of your modes at a specific time (e.g., t_idx)
bz_00_real = bz[t_idx, :, mode_00].real

# Need both real and imag for modes where m > 0 or n > 0
bz_11_real = bz[t_idx, :, mode_11].real
bz_11_imag = bz[t_idx, :, mode_11].imag

# 3. Initialize the empty physical field array
Bz_physical = np.zeros_like(R_grid)

# 4. Add the m=0, n=0 mode (Equilibrium)
# Since m=0 and n=0, cos(0)=1 and sin(0)=0. The physical field is just the real part.
Bz_physical += bz_00_real[np.newaxis, :]

# 5. Add the m=1, n=1 mode (Tearing Mode)
# At toroidal angle phi = 0, the angle term simplifies to just m*theta (which is 1*theta)
Bz_physical += (bz_11_real[np.newaxis, :] * np.cos(Theta_grid) - 
                bz_11_imag[np.newaxis, :] * np.sin(Theta_grid))

# 6. (Optional) Loop over ALL modes in your spectrum to get the complete physical field
# For a full reconstruction, you would loop through your entire 'bz' array, 
# extracting the m and n for each index, and adding them to Bz_physical just like step 5.
import numpy as np
import matplotlib.pyplot as plt
# ==========================================================
# 1. Setup Grid and Phase Indices
# ==========================================================
R = 4.0
tolerance = 1e-10

idx_start = 550   
idx_peak = 690    
idx_end = 740     
time_indices = [idx_start, idx_peak, idx_end]
phase_names = ['Linear Phase', 'Pre-Crash', 'Post-Crash']

mode_00 = janz(0, 0, input_folder)
mode_11 = janz(1, 1, input_folder)

# *** ADD THESE LINES TO DEFINE X AND Y ***
theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta)
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# ==========================================================
# 2. Plotting 2x3 Grid of 2D Contours
# ==========================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for col, (t_idx, name) in enumerate(zip(time_indices, phase_names)):
    # ... rest of your loop continues here ...
    # ------------------------------------------------------
    # A. Equilibrium (m=0, n=0) Current Profiles
    # ------------------------------------------------------
    bz_00 = bz[t_idx, :, mode_00].real
    bt_00 = bt[t_idx, :, mode_00].real

    dbt00_dr = np.gradient(bt_00, r2)
    b_theta_over_r_00 = np.zeros_like(r2)
    
    safe_mask = r2 > tolerance
    b_theta_over_r_00[safe_mask] = bt_00[safe_mask] / r2[safe_mask]
    b_theta_over_r_00[~safe_mask] = dbt00_dr[~safe_mask]

    jz_00 = b_theta_over_r_00 + dbt00_dr
    jt_00 = -np.gradient(bz_00, r2)

    # ------------------------------------------------------
    # B. Perturbation (m=1, n=1) Current Profiles
    # ------------------------------------------------------
    # Br is on the r1 grid; interpolate to r2 to align coordinates
    br_11 = np.interp(r2, r1, br[t_idx, :, mode_11])
    bt_11 = bt[t_idx, :, mode_11]
    bz_11 = bz[t_idx, :, mode_11]

    dbt11_dr = np.gradient(bt_11, r2)
    dbz11_dr = np.gradient(bz_11, r2)

    # J_theta (m=1, n=1) = -i * (n/R) * B_r - d(B_z)/dr
    jt_11 = -1j * (1.0 / R) * br_11 - dbz11_dr

    # J_z (m=1, n=1) = (1/r)*d(r*B_t)/dr - (i*m/r)*B_r
    # Using L'Hopital at r=0: lim (Bt - i*Br)/r = d(Bt)/dr - i*d(Br)/dr
    dbr11_dr = np.gradient(br_11, r2)
    term_over_r = np.zeros_like(r2, dtype=complex)
    term_over_r[safe_mask] = (bt_11[safe_mask] - 1j * br_11[safe_mask]) / r2[safe_mask]
    term_over_r[~safe_mask] = dbt11_dr[~safe_mask] - 1j * dbr11_dr[~safe_mask]
    
    jz_11 = term_over_r + dbt11_dr

    # ------------------------------------------------------
    # C. Transform back to 2D Ordinary Space (phi = 0)
    # ------------------------------------------------------
    # J(r, theta) = J_00(r) + Re[J_11] * cos(theta) - Im[J_11] * sin(theta)
    Jz_2D = (jz_00[np.newaxis, :] + 
             jz_11.real[np.newaxis, :] * np.cos(Theta_grid) - 
             jz_11.imag[np.newaxis, :] * np.sin(Theta_grid))

    Jt_2D = (jt_00[np.newaxis, :] + 
             jt_11.real[np.newaxis, :] * np.cos(Theta_grid) - 
             jt_11.imag[np.newaxis, :] * np.sin(Theta_grid))

    # ------------------------------------------------------
    # D. Plotting J_z (Top Row)
    # ------------------------------------------------------
    ax_z = axes[0, col]
    c_z = ax_z.contourf(X, Y, Jz_2D, levels=50, cmap='plasma')
    ax_z.set_title(f"$J_z$: {name}\n($t = {t[t_idx]:.0f}$)", fontsize=13)
    ax_z.set_xlabel('X')
    ax_z.set_ylabel('Y')
    ax_z.set_aspect('equal')
    fig.colorbar(c_z, ax=ax_z, fraction=0.046, pad=0.04)

    # ------------------------------------------------------
    # E. Plotting J_theta (Bottom Row)
    # ------------------------------------------------------
    ax_t = axes[1, col]
    c_t = ax_t.contourf(X, Y, Jt_2D, levels=50, cmap='viridis')
    ax_t.set_title(f"$J_\\theta$: {name}\n($t = {t[t_idx]:.0f}$)", fontsize=13)
    ax_t.set_xlabel('X')
    ax_t.set_ylabel('Y')
    ax_t.set_aspect('equal')
    fig.colorbar(c_t, ax=ax_t, fraction=0.046, pad=0.04)

plt.tight_layout(h_pad=3.0, w_pad=1.0)
plt.show()
# ==========================================================
# NEW CODE: Hunting for Current Sheets (Perturbed Current)
# ==========================================================

# FIX: Define the colors and line styles for the three phases
colors = ['blue', 'orange', 'red'] 
line_styles = [':', '--', '-']

plt.figure(figsize=(10, 6))

for i, t_idx in enumerate(time_indices):
    # Extract just the m=1, n=1 perturbation fields
    br_11 = np.interp(r2, r1, br[t_idx, :, mode_11])
    bt_11 = bt[t_idx, :, mode_11]
    
    # Calculate derivatives
    dbr11_dr = np.gradient(br_11, r2)
    dbt11_dr = np.gradient(bt_11, r2)
    
    # Calculate J_z for m=1, n=1
    # J_z = (1/r)*d(r*B_t)/dr - (i*m/r)*B_r
    term_over_r = np.zeros_like(r2, dtype=complex)
    safe_mask = r2 > tolerance
    
    term_over_r[safe_mask] = (bt_11[safe_mask] - 1j * br_11[safe_mask]) / r2[safe_mask]
    # L'Hopital's for r=0
    term_over_r[~safe_mask] = dbt11_dr[~safe_mask] - 1j * dbr11_dr[~safe_mask]
    
    jz_11 = term_over_r + dbt11_dr
    
    # Plot the magnitude (envelope) of the perturbed current
    plt.plot(r2, np.abs(jz_11), color=colors[i], linestyle=line_styles[i], 
             label=f"{phase_names[i]} (t={t[t_idx]:.0f})")

plt.axvline(0.4, color='gray', linestyle=':', alpha=0.7, label="Approx q=1 surface")

plt.xlabel('Radius (r)')
plt.ylabel(r'Perturbed Current Magnitude $|J_{z, 1,1}|$')
plt.title('Hunting for Current Sheets: Perturbed Axial Current')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()