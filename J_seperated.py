import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import readsav

input_folder = "C:/data.specly/"

def janz(m, n, directory):
    if not directory.endswith('/'): directory += '/'
    spectr_arrays = readsav(directory + 'spectrum.sav')
    nanf, nz = spectr_arrays['NANF'].view(), spectr_arrays['NZ'].view()
    j = 0
    if m == 0 and n == 0: return int(j)
    for im in range(m): j += nz[im]
    return int(j + n - nanf[m] + 1)

print("Loading Tokamak data...")
fields = readsav(input_folder + 'modeprofiles_10000.sav')
t_array = fields['prot'].view()

br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))
r1, r2 = fields['pror1'].view(), fields['pror2'].view() 

spectr = readsav(input_folder + 'spectrum.sav')
mm_array, nanf_array, nz_array = spectr['MM'].view(), spectr['NANF'].view(), spectr['NZ'].view()

R_major = 4.0
z_position = 0.0
tolerance = 1e-10

INCLUDE_BACKGROUND = True
t_start, t_end, step = 650, 750, 2 
frame_indices = list(range(t_start, t_end, step))

theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta, indexing='ij') 
X, Y = R_grid * np.cos(Theta_grid), R_grid * np.sin(Theta_grid)

def compute_components(t_idx):
    Jr, Jt, Jz = np.zeros_like(R_grid, dtype=float), np.zeros_like(R_grid, dtype=float), np.zeros_like(R_grid, dtype=float)
    
    for i in range(len(mm_array)):
        m, n_start = int(mm_array[i]), int(nanf_array[i])
        for n_step in range(int(nz_array[i])):
            n = n_start + n_step
            if not INCLUDE_BACKGROUND and m == 0 and n == 0: continue
                
            mode_idx = janz(m, n, input_folder)
            if mode_idx >= bz.shape[2]: continue
            
            br_c = np.interp(r2, r1, br[t_idx, :, mode_idx].real) + 1j * np.interp(r2, r1, br[t_idx, :, mode_idx].imag)
            bt_c, bz_c = bt[t_idx, :, mode_idx], bz[t_idx, :, mode_idx]
            
            dbt_dr, dbz_dr, dbr_dr = np.gradient(bt_c, r2), np.gradient(bz_c, r2), np.gradient(br_c, r2)
            bt_over_r, bz_over_r, br_over_r = np.zeros_like(bt_c), np.zeros_like(bz_c), np.zeros_like(br_c)
            
            safe = r2 > tolerance
            np.divide(bt_c, r2, out=bt_over_r, where=safe)
            np.divide(bz_c, r2, out=bz_over_r, where=safe)
            np.divide(br_c, r2, out=br_over_r, where=safe)
            
            zero = r2 <= tolerance
            bt_over_r[zero], bz_over_r[zero], br_over_r[zero] = dbt_dr[zero], dbz_dr[zero], dbr_dr[zero]
            
            k = n / R_major
            jr_1D = 1j * m * bz_over_r - 1j * k * bt_c
            jt_1D = 1j * k * br_c - dbz_dr
            jz_1D = bt_over_r + dbt_dr - 1j * m * br_over_r
            
            phase = m * Theta_grid + k * z_position
            Jr += (jr_1D.real[:, np.newaxis] * np.cos(phase) - jr_1D.imag[:, np.newaxis] * np.sin(phase))
            Jt += (jt_1D.real[:, np.newaxis] * np.cos(phase) - jt_1D.imag[:, np.newaxis] * np.sin(phase))
            Jz += (jz_1D.real[:, np.newaxis] * np.cos(phase) - jz_1D.imag[:, np.newaxis] * np.sin(phase))
                
    return Jr, Jt, Jz

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

print("Pre-scanning peak crash frame to lock symmetric color limits...")
Jr_p, Jt_p, Jz_p = compute_components(728)
vmax_r = np.max(np.abs(Jr_p)) * 1.05
vmax_t = np.max(np.abs(Jt_p)) * 1.05
vmax_z = np.max(np.abs(Jz_p)) * 1.05

Jr0, Jt0, Jz0 = compute_components(frame_indices[0])
mesh_r = axes[0].pcolormesh(X, Y, Jr0, cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r, shading='gouraud')
mesh_t = axes[1].pcolormesh(X, Y, Jt0, cmap='RdBu_r', vmin=-vmax_t, vmax=vmax_t, shading='gouraud')
mesh_z = axes[2].pcolormesh(X, Y, Jz0, cmap='RdBu_r', vmin=-vmax_z, vmax=vmax_z, shading='gouraud')

fig.colorbar(mesh_r, ax=axes[0], orientation='horizontal', pad=0.1, label='Radial Current ($J_r$)')
fig.colorbar(mesh_t, ax=axes[1], orientation='horizontal', pad=0.1, label='Poloidal Current ($J_\\theta$)')
fig.colorbar(mesh_z, ax=axes[2], orientation='horizontal', pad=0.1, label='Axial Current ($J_z$)')

fig.suptitle("Tokamak Current Density Components", fontsize=16)

def update(frame_idx):
    Jr, Jt, Jz = compute_components(frame_idx)
    mesh_r.set_array(Jr.ravel())
    mesh_t.set_array(Jt.ravel())
    mesh_z.set_array(Jz.ravel())
    fig.suptitle(f"Tokamak Current Density Components (Time: {t_array[frame_idx]:.2f})", fontsize=16)
    return mesh_r, mesh_t, mesh_z

ani = FuncAnimation(fig, update, frames=frame_indices, blit=False)
output_file = "C:/Users/mobin/Desktop/Tokamak_J_Components.gif"
print(f"Saving to {output_file}...")
ani.save(output_file, writer='pillow', fps=10)
print("SUCCESS!")