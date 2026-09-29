import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

input_folder = "C:/data.specly/RFP/"

def janz(m, n, directory):
    if not directory.endswith('/'):
        directory += '/'
    spectr = readsav(directory + 'spectrum.sav')
    nanf, nz = spectr['NANF'].view(), spectr['NZ'].view()
    j = 0
    if m == 0 and n == 0: return int(j)
    for im in range(m): j += nz[im]
    return int(j + n - nanf[m] + 1)

# ==========================================
# 1. Load Data
# ==========================================
print("Loading RFP data...")
fields = readsav(input_folder + 'modeprofiles.sav')
t = fields['t'].view()

br = np.transpose(fields['br'].view(), (2, 1, 0))
bt = np.transpose(fields['bt'].view(), (2, 1, 0))
bz = np.transpose(fields['bz'].view(), (2, 1, 0))
r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

mode_00 = janz(0, 0, input_folder)
m_val, n_val = 1, -6
mode_dom = janz(m_val, n_val, input_folder)

R_major = 4.0
tolerance = 1e-10

# ==========================================
# 2. Spatiotemporal Heatmaps (q and J_z)
# ==========================================
print("Calculating Heatmaps...")
bz_00_all = bz[:, :, mode_00].real
bt_00_all = bt[:, :, mode_00].real

dbt_dr_all = np.gradient(bt_00_all, r2, axis=1)

# Safety Factor (q)
q_all = np.zeros_like(bz_00_all)
safe_mask = r2 > tolerance
q_all[:, safe_mask] = (r2[safe_mask] * bz_00_all[:, safe_mask]) / (R_major * bt_00_all[:, safe_mask])
zero_mask = r2 <= tolerance
q_all[:, zero_mask] = bz_00_all[:, zero_mask] / (R_major * dbt_dr_all[:, zero_mask])

# Axial Current (J_z)
b_theta_over_r_all = np.zeros_like(bt_00_all)
b_theta_over_r_all[:, safe_mask] = bt_00_all[:, safe_mask] / r2[safe_mask]
b_theta_over_r_all[:, zero_mask] = dbt_dr_all[:, zero_mask]
jz_all = b_theta_over_r_all + dbt_dr_all

T_grid, R_grid = np.meshgrid(t, r2)

# Plot q Heatmap
plt.figure(figsize=(12, 5))
heatmap_q = plt.pcolormesh(T_grid, R_grid, q_all.T, cmap='viridis', shading='nearest')
contour = plt.contour(T_grid, R_grid, q_all.T, levels=[0.0], colors='white', linewidths=2)
plt.clabel(contour, inline=True, fontsize=12, fmt="q=0 (Reversal)")
plt.colorbar(heatmap_q, label='Safety Factor (q)')
plt.xlabel('Time (t)')
plt.ylabel('Radius (r)')
plt.title('RFP Spatiotemporal Evolution of the Safety Factor')
plt.show(block=False)

# Plot J_z Heatmap
plt.figure(figsize=(12, 5))
heatmap_jz = plt.pcolormesh(T_grid, R_grid, jz_all.T, cmap='plasma', shading='nearest')
plt.colorbar(heatmap_jz, label=r'Axial Current Density ($J_z$)')
plt.xlabel('Time (t)')
plt.ylabel('Radius (r)')
plt.title(r'RFP Spatiotemporal Evolution of Axial Current Density ($J_z$)')
plt.show(block=False)

# ==========================================
# 3. 2x3 Grid of 2D Contours (Total Field)
# ==========================================
print("Calculating 2D Contours...")
theta = np.linspace(0, 2 * np.pi, 200)
R_grid2D, Theta_grid2D = np.meshgrid(r2, theta)
X = R_grid2D * np.cos(Theta_grid2D)
Y = R_grid2D * np.sin(Theta_grid2D)

time_indices = [0, len(t)//2, -1]
phase_names = ['Startup', 'Relaxation', 'Mature RFP']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for col, (t_idx, name) in enumerate(zip(time_indices, phase_names)):
    # Equilibrium
    bz_00 = bz[t_idx, :, mode_00].real
    bt_00 = bt[t_idx, :, mode_00].real
    dbt00_dr = np.gradient(bt_00, r2)
    b_theta_over_r_00 = np.zeros_like(r2)
    b_theta_over_r_00[safe_mask] = bt_00[safe_mask] / r2[safe_mask]
    b_theta_over_r_00[~safe_mask] = dbt00_dr[~safe_mask]
    jz_00 = b_theta_over_r_00 + dbt00_dr
    jt_00 = -np.gradient(bz_00, r2)

    # Perturbation
    br_dom = np.interp(r2, r1, br[t_idx, :, mode_dom])
    bt_dom = bt[t_idx, :, mode_dom]
    bz_dom = bz[t_idx, :, mode_dom]

    dbt_dom_dr = np.gradient(bt_dom, r2)
    dbz_dom_dr = np.gradient(bz_dom, r2)
    dbr_dom_dr = np.gradient(br_dom, r2)

    jt_dom = -1j * (n_val / R_major) * br_dom - dbz_dom_dr

    term_over_r = np.zeros_like(r2, dtype=complex)
    term_over_r[safe_mask] = (bt_dom[safe_mask] - 1j * m_val * br_dom[safe_mask]) / r2[safe_mask]
    term_over_r[~safe_mask] = dbt_dom_dr[~safe_mask] - 1j * m_val * dbr_dom_dr[~safe_mask]
    jz_dom = term_over_r + dbt_dom_dr

    # 2D Projection (phi = 0)
    Jz_2D = (jz_00[np.newaxis, :] + 
             jz_dom.real[np.newaxis, :] * np.cos(m_val * Theta_grid2D) - 
             jz_dom.imag[np.newaxis, :] * np.sin(m_val * Theta_grid2D))

    Jt_2D = (jt_00[np.newaxis, :] + 
             jt_dom.real[np.newaxis, :] * np.cos(m_val * Theta_grid2D) - 
             jt_dom.imag[np.newaxis, :] * np.sin(m_val * Theta_grid2D))

    # Plot J_z
    ax_z = axes[0, col]
    c_z = ax_z.contourf(X, Y, Jz_2D, levels=50, cmap='plasma')
    ax_z.set_title(f"$J_z$: {name}\n($t = {t[t_idx]:.0f}$)", fontsize=13)
    ax_z.set_aspect('equal')
    fig.colorbar(c_z, ax=ax_z, fraction=0.046, pad=0.04)

    # Plot J_theta
    ax_t = axes[1, col]
    c_t = ax_t.contourf(X, Y, Jt_2D, levels=50, cmap='viridis')
    ax_t.set_title(f"$J_\\theta$: {name}\n($t = {t[t_idx]:.0f}$)", fontsize=13)
    ax_t.set_aspect('equal')
    fig.colorbar(c_t, ax=ax_t, fraction=0.046, pad=0.04)

plt.tight_layout(h_pad=3.0, w_pad=1.0)
plt.show(block=False)

# ==========================================================
# 4. Hunting for Current Sheets (Perturbed Current)
# ==========================================================
print("Calculating 1D Perturbations...")
colors = ['blue', 'orange', 'red'] 
line_styles = [':', '--', '-']

plt.figure(figsize=(10, 6))

for i, t_idx in enumerate(time_indices):
    br_dom = np.interp(r2, r1, br[t_idx, :, mode_dom])
    bt_dom = bt[t_idx, :, mode_dom]
    
    dbr_dom_dr = np.gradient(br_dom, r2)
    dbt_dom_dr = np.gradient(bt_dom, r2)
    
    term_over_r = np.zeros_like(r2, dtype=complex)
    term_over_r[safe_mask] = (bt_dom[safe_mask] - 1j * m_val * br_dom[safe_mask]) / r2[safe_mask]
    term_over_r[~safe_mask] = dbt_dom_dr[~safe_mask] - 1j * m_val * dbr_dom_dr[~safe_mask]
    
    jz_dom = term_over_r + dbt_dom_dr
    
    plt.plot(r2, np.abs(jz_dom), color=colors[i], linestyle=line_styles[i], 
             label=f"{phase_names[i]} (t={t[t_idx]:.0f})")

# Estimate the reversal surface location from the final time step
q_mature = q_all[-1, :]
reversal_idx = np.argmin(np.abs(q_mature))
reversal_radius = r2[reversal_idx]

plt.axvline(reversal_radius, color='gray', linestyle=':', alpha=0.7, label=f"Approx q=0 surface (r={reversal_radius:.2f})")

plt.xlabel('Radius (r)')
plt.ylabel(f'Perturbed Current Magnitude $|J_{{z, {m_val}, {n_val}}}|$')
plt.title(f'RFP Current Sheet Hunt: Mode ({m_val}, {n_val})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.show()