import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

input_folder = "C:/data.specly/"

def janz(m, n, directory):
    if not directory.endswith('/'): directory += '/'
    spectr = readsav(directory + 'spectrum.sav')
    nanf, nz = spectr['NANF'].view(), spectr['NZ'].view()
    j = 0
    if m == 0 and n == 0: return int(j)
    for im in range(m): j += nz[im]
    return int(j + n - nanf[m] + 1)

# ==========================================
# 1. Load Tokamak Data
# ==========================================
print("Loading Tokamak data...")
fields = readsav(input_folder + 'modeprofiles_10000.sav')

t_array = fields['prot'].view()
br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))

r1, r2 = fields['pror1'].view(), fields['pror2'].view() 

m_val, n_val = 1, 1
mode_idx = janz(m_val, n_val, input_folder)

R_major = 4.0
k = n_val / R_major
tolerance = 1e-10

# ==========================================
# 2. Calculate the 2D Gradient Array (Time x Radius)
# ==========================================
print(f"Hunting Current Sheets via Radial Gradient for Mode ({m_val}, {n_val})...")

num_times = len(t_array)
num_radii = len(r2)
dJ_mag_2D = np.zeros((num_times, num_radii))

safe_mask = r2 > tolerance
zero_mask = r2 <= tolerance

for t_idx in range(num_times):
    br_raw = br[t_idx, :, mode_idx]
    br_c = np.interp(r2, r1, br_raw.real) + 1j * np.interp(r2, r1, br_raw.imag)
    bt_c = bt[t_idx, :, mode_idx]
    bz_c = bz[t_idx, :, mode_idx]
    
    dbr_dr, dbt_dr, dbz_dr = np.gradient(br_c, r2), np.gradient(bt_c, r2), np.gradient(bz_c, r2)
    
    br_over_r, bt_over_r, bz_over_r = np.zeros_like(br_c), np.zeros_like(bt_c), np.zeros_like(bz_c)
    br_over_r[safe_mask] = br_c[safe_mask] / r2[safe_mask]
    bt_over_r[safe_mask] = bt_c[safe_mask] / r2[safe_mask]
    bz_over_r[safe_mask] = bz_c[safe_mask] / r2[safe_mask]
    
    br_over_r[zero_mask], bt_over_r[zero_mask], bz_over_r[zero_mask] = dbr_dr[zero_mask], dbt_dr[zero_mask], dbz_dr[zero_mask]
    
    jr_c = 1j * m_val * bz_over_r - 1j * k * bt_c
    jt_c = 1j * k * br_c - dbz_dr
    jz_c = bt_over_r + dbt_dr - 1j * m_val * br_over_r
    
    # EXPLICIT MATH: J_{r, \theta, z} vector -> \partial_r -> |...|
    J_rtz = np.array([jr_c, jt_c, jz_c])
    dJ_rtz_dr = np.gradient(J_rtz, r2, axis=1)
    abs_dJ_rtz_dr = np.linalg.norm(dJ_rtz_dr, axis=0)
    
    dJ_mag_2D[t_idx, :] = abs_dJ_rtz_dr

# EXPLICIT MATH: max_r(...)
max_dJ_over_time = np.max(dJ_mag_2D, axis=1)

# ==========================================
# 3. Plotting
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

T_grid, R_grid = np.meshgrid(t_array, r2, indexing='ij')

# FIX: Capping the color scale at the 99.5th percentile to reveal darker structures
vmax_val = np.percentile(dJ_mag_2D, 99.5)
heatmap = ax1.pcolormesh(T_grid, R_grid, dJ_mag_2D, cmap='turbo', vmax=vmax_val, shading='nearest')
fig.colorbar(heatmap, ax=ax1, label=rf'Gradient Magnitude $|\partial_r \tilde{{J}}^{{{m_val},{n_val}}}|$')

ax1.set_title(f'Tokamak Current Sheet Locator: Radial Gradient of Mode ({m_val}, {n_val})')
ax1.set_ylabel('Radius (r)')
ax1.set_xticklabels([]) 

ax2.plot(t_array, max_dJ_over_time, color='purple', linewidth=2)
ax2.fill_between(t_array, max_dJ_over_time, color='purple', alpha=0.2)

ax2.set_title(rf'Peak Sheet Intensity: $\max_r(|\partial_r \tilde{{J}}^{{{m_val},{n_val}}}(r, t)|)$')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Max Gradient Magnitude')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(t_array[0], t_array[-1])

plt.tight_layout()
plt.show()