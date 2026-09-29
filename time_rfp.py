import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import readsav

input_folder = "C:/data.specly/RFP/"

def janz(m, n, directory):
    if not directory.endswith('/'):
        directory += '/'
    spectr_arrays = readsav(directory + 'spectrum.sav')
    nanf, nz = spectr_arrays['NANF'].view(), spectr_arrays['NZ'].view()
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
mode_pert = janz(m_val, n_val, input_folder)

R_major = 4.0
k = n_val / R_major
tolerance = 1e-10

# ==========================================
# 2. Setup 2D Grid
# ==========================================
theta = np.linspace(0, 2 * np.pi, 150)
# indexing='ij' ensures shapes align as (radius, theta)
R_grid, Theta_grid = np.meshgrid(r2, theta, indexing='ij') 
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

safe_mask = r2 > tolerance
zero_mask = r2 <= tolerance

# ==========================================
# 3. Setup Animation Figure
# ==========================================
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_aspect('equal')
ax.set_xlabel('X (r * cos θ)')
ax.set_ylabel('Y (r * sin θ)')

# We need a fixed colorbar scale so the animation doesn't flicker.
# Based on your previous RFP 1D plots, core Jz is around 3.2. We set vmax slightly higher.
vmax_fixed = 4.0 
vmin_fixed = 0.0
levels = np.linspace(vmin_fixed, vmax_fixed, 60)

# Create a dummy contour to initialize the colorbar
dummy_contour = ax.contourf(X, Y, np.zeros_like(X), levels=levels, cmap='turbo')
cbar = fig.colorbar(dummy_contour, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Total Current Magnitude ($|J_{tot}|$)')

def compute_jtot(t_idx):
    # A. Equilibrium Current (m=0, n=0)
    bz_00 = bz[t_idx, :, mode_00].real
    bt_00 = bt[t_idx, :, mode_00].real
    
    dbt00_dr = np.gradient(bt_00, r2)
    dbz00_dr = np.gradient(bz_00, r2)
    
    b_theta_over_r_00 = np.zeros_like(r2)
    b_theta_over_r_00[safe_mask] = bt_00[safe_mask] / r2[safe_mask]
    b_theta_over_r_00[zero_mask] = dbt00_dr[zero_mask]
    
    jz_00 = b_theta_over_r_00 + dbt00_dr
    jt_00 = -dbz00_dr
    jr_00 = np.zeros_like(r2)
    
    # B. Perturbation Current (m=1, n=-6)
    br_pert = np.interp(r2, r1, br[t_idx, :, mode_pert])
    bt_pert = bt[t_idx, :, mode_pert]
    bz_pert = bz[t_idx, :, mode_pert]
    
    dbr_pert_dr = np.gradient(br_pert, r2)
    dbt_pert_dr = np.gradient(bt_pert, r2)
    dbz_pert_dr = np.gradient(bz_pert, r2)
    
    bz_over_r_pert = np.zeros_like(bz_pert)
    bt_over_r_pert = np.zeros_like(bt_pert)
    br_over_r_pert = np.zeros_like(br_pert)
    
    bz_over_r_pert[safe_mask] = bz_pert[safe_mask] / r2[safe_mask]
    bt_over_r_pert[safe_mask] = bt_pert[safe_mask] / r2[safe_mask]
    br_over_r_pert[safe_mask] = br_pert[safe_mask] / r2[safe_mask]
    
    bz_over_r_pert[zero_mask] = dbz_pert_dr[zero_mask]
    bt_over_r_pert[zero_mask] = dbt_pert_dr[zero_mask]
    br_over_r_pert[zero_mask] = dbr_pert_dr[zero_mask]
    
    jr_pert_1D = 1j * m_val * bz_over_r_pert - 1j * k * bt_pert
    jt_pert_1D = 1j * k * br_pert - dbz_pert_dr
    jz_pert_1D = bt_over_r_pert + dbt_pert_dr - 1j * m_val * br_over_r_pert
    
    # C. Map to 2D and Combine
    phase = m_val * Theta_grid # z=0 slice
    
    Jr_2D = jr_00[:, np.newaxis] + jr_pert_1D.real[:, np.newaxis] * np.cos(phase) - jr_pert_1D.imag[:, np.newaxis] * np.sin(phase)
    Jt_2D = jt_00[:, np.newaxis] + jt_pert_1D.real[:, np.newaxis] * np.cos(phase) - jt_pert_1D.imag[:, np.newaxis] * np.sin(phase)
    Jz_2D = jz_00[:, np.newaxis] + jz_pert_1D.real[:, np.newaxis] * np.cos(phase) - jz_pert_1D.imag[:, np.newaxis] * np.sin(phase)
    
    J_tot_mag = np.sqrt(Jr_2D**2 + Jt_2D**2 + Jz_2D**2)
    return J_tot_mag

def update(frame):
    ax.clear()
    J_mag = compute_jtot(frame)
    
    # Re-draw formatting
    ax.set_aspect('equal')
    ax.set_xlabel('X (r * cos θ)')
    ax.set_ylabel('Y (r * sin θ)')
    ax.set_title(f"RFP Total Current Evolution (Mode 1,6 Included)\nTime: {t[frame]:.2f} (Index: {frame})")
    
    # Use turbo colormap to match your reference image
    cont = ax.contourf(X, Y, J_mag, levels=levels, cmap='turbo', extend='max')
    return cont,

# ==========================================
# 4. Generate Animation
# ==========================================
print("Generating Animation... this may take a minute depending on data size.")

# Create an array of frames. Using every 2nd frame to speed up rendering.
# Adjust the slice [::2] if you want it faster/slower.
frames_to_render = np.arange(0, len(t), 2)

ani = animation.FuncAnimation(fig, update, frames=frames_to_render, blit=False, repeat=True)

# Save as a GIF to your Desktop
ani.save("C:/Users/mobin/Desktop/RFP_Current_Evolution.gif", writer='pillow', fps=15)
print("Saved to Desktop!")

# plt.show() # You can comment this out so it doesn't try to open the popup window