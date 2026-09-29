import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import readsav

input_folder = "C:/data.specly/"

def janz(m, n, directory):
    spectr_arrays = readsav(directory + 'spectrum.sav')
    nanf = spectr_arrays['NANF'].view()
    nz = spectr_arrays['NZ'].view()
    j = 0
    if m == 0 and n == 0: return int(j)
    for im in range(m): j = j + nz[im]
    return int(j + n - nanf[m] + 1)

# ==========================================
# 1. Load Data & Generate ALL Modes
# ==========================================
print("Loading data...")
fields = readsav(input_folder + 'modeprofiles_10000.sav')
t_array = fields['prot'].view()

br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))

r1 = fields['pror1'].view() 
r2 = fields['pror2'].view() 

spectr = readsav(input_folder + 'spectrum.sav')
mm_array = spectr['MM'].view()
nanf_array = spectr['NANF'].view()
nz_array = spectr['NZ'].view()

R_major = 4.0
tolerance = 1e-10

# Animation Time Window
t_start = 650
t_end = 750
step = 2
frame_indices = range(t_start, t_end, step)

# Build a list of EVERY mode in the simulation (including 0,0 background)
all_modes = []
for i in range(len(mm_array)):
    m = int(mm_array[i])
    n_start = int(nanf_array[i])
    for n_step in range(int(nz_array[i])):
        all_modes.append((m, n_start + n_step))

print(f"Discovered {len(all_modes)} total modes to simulate.")

# ==========================================
# 2. Physics Engine
# ==========================================
def compute_1D_J_complex(t_idx, m, n):
    mode_idx = janz(m, n, input_folder)
    if mode_idx >= bz.shape[2]: return None
    
    br_raw = br[t_idx, :, mode_idx]      
    bt_complex = bt[t_idx, :, mode_idx]  
    bz_complex = bz[t_idx, :, mode_idx]
    
    br_complex = np.interp(r2, r1, br_raw.real) + 1j * np.interp(r2, r1, br_raw.imag)
    
    dbt_dr = np.gradient(bt_complex, r2)
    dbz_dr = np.gradient(bz_complex, r2)
    
    bt_over_r = np.zeros_like(bt_complex)
    br_over_r = np.zeros_like(br_complex)
    bz_over_r = np.zeros_like(bz_complex)
    safe_mask = r2 > tolerance
    
    np.divide(bt_complex, r2, out=bt_over_r, where=safe_mask)
    np.divide(br_complex, r2, out=br_over_r, where=safe_mask)
    np.divide(bz_complex, r2, out=bz_over_r, where=safe_mask)
    
    zero_mask = r2 <= tolerance
    bt_over_r[zero_mask] = dbt_dr[zero_mask]
    bz_over_r[zero_mask] = dbz_dr[zero_mask]
    br_over_r[zero_mask] = np.gradient(br_complex, r2)[zero_mask] 
    
    k = n / R_major
    jr_1D = 1j * m * bz_over_r - 1j * k * bt_complex
    jt_1D = 1j * k * br_complex - dbz_dr
    jz_1D = bt_over_r + dbt_dr - 1j * m * br_over_r
    
    return jr_1D, jt_1D, jz_1D

# ==========================================
# 3. Setup Spatial Grids
# ==========================================
theta_1d = np.linspace(0, 2 * np.pi, 150)
z_1d = np.linspace(0, 2 * np.pi * R_major, 150)
r_idx = np.argmin(np.abs(r2 - 0.42)) # q-surface index for Panel 3

R_grid1, T_grid1 = np.meshgrid(r2, theta_1d, indexing='ij')
X1, Y1 = R_grid1 * np.cos(T_grid1), R_grid1 * np.sin(T_grid1)
Z_grid2, R_grid2 = np.meshgrid(z_1d, r2, indexing='ij')
Z_grid3, T_grid3 = np.meshgrid(z_1d, theta_1d, indexing='ij')

def calculate_frame_slices(t_idx):
    Jr1, Jt1, Jz1 = np.zeros_like(R_grid1), np.zeros_like(R_grid1), np.zeros_like(R_grid1)
    Jr2, Jt2, Jz2 = np.zeros_like(Z_grid2), np.zeros_like(Z_grid2), np.zeros_like(Z_grid2)
    Jr3, Jt3, Jz3 = np.zeros_like(Z_grid3), np.zeros_like(Z_grid3), np.zeros_like(Z_grid3)
    
    for (m, n) in all_modes:
        res = compute_1D_J_complex(t_idx, m, n)
        if res:
            jr_c, jt_c, jz_c = res
            
            p1 = m * T_grid1
            Jr1 += jr_c.real[:, np.newaxis] * np.cos(p1) - jr_c.imag[:, np.newaxis] * np.sin(p1)
            Jt1 += jt_c.real[:, np.newaxis] * np.cos(p1) - jt_c.imag[:, np.newaxis] * np.sin(p1)
            Jz1 += jz_c.real[:, np.newaxis] * np.cos(p1) - jz_c.imag[:, np.newaxis] * np.sin(p1)
            
            p2 = (n / R_major) * Z_grid2
            Jr2 += jr_c.real[np.newaxis, :] * np.cos(p2) - jr_c.imag[np.newaxis, :] * np.sin(p2)
            Jt2 += jt_c.real[np.newaxis, :] * np.cos(p2) - jt_c.imag[np.newaxis, :] * np.sin(p2)
            Jz2 += jz_c.real[np.newaxis, :] * np.cos(p2) - jz_c.imag[np.newaxis, :] * np.sin(p2)
            
            p3 = m * T_grid3 + (n / R_major) * Z_grid3
            Jr3 += jr_c.real[r_idx] * np.cos(p3) - jr_c.imag[r_idx] * np.sin(p3)
            Jt3 += jt_c.real[r_idx] * np.cos(p3) - jt_c.imag[r_idx] * np.sin(p3)
            Jz3 += jz_c.real[r_idx] * np.cos(p3) - jz_c.imag[r_idx] * np.sin(p3)

    return (np.sqrt(Jr1**2 + Jt1**2 + Jz1**2), 
            np.sqrt(Jr2**2 + Jt2**2 + Jz2**2), 
            np.sqrt(Jr3**2 + Jt3**2 + Jz3**2))

# ==========================================
# 4. Pre-compute Panel 4 (Hovmöller)
# ==========================================
print("Pre-computing Hovmöller Diagram (This calculates every mode, takes a few seconds)...")
time_subset = t_array[t_start:t_end]
Time_grid4, R_grid4 = np.meshgrid(time_subset, r2, indexing='ij')
Jr4, Jt4, Jz4 = np.zeros_like(R_grid4), np.zeros_like(R_grid4), np.zeros_like(R_grid4)

for offset, t_idx in enumerate(range(t_start, t_end)):
    for (m, n) in all_modes:
        res = compute_1D_J_complex(t_idx, m, n)
        if res:
            jr_c, jt_c, jz_c = res
            Jr4[offset, :] += jr_c.real
            Jt4[offset, :] += jt_c.real
            Jz4[offset, :] += jz_c.real
Mag4 = np.sqrt(Jr4**2 + Jt4**2 + Jz4**2)

# ==========================================
# 5. Animation Setup (WITH COLORBARS)
# ==========================================
# Increased width slightly to fit the 4 colorbars cleanly!
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
plt.subplots_adjust(wspace=0.35, hspace=0.3)

print("Scanning peak crash for locked color scales...")
M1_max, M2_max, M3_max = calculate_frame_slices(710)
vmax1, vmax2, vmax3 = np.max(M1_max)*1.05, np.max(M2_max)*1.05, np.max(M3_max)*1.05
vmax4 = np.max(Mag4)*1.05

print("Rendering initial frames...")
M1, M2, M3 = calculate_frame_slices(frame_indices[0])

# Panel 1
mesh1 = axes[0, 0].pcolormesh(X1, Y1, M1, cmap='turbo', vmin=0, vmax=vmax1, shading='gouraud')
plt.colorbar(mesh1, ax=axes[0, 0], fraction=0.046, pad=0.04, label=r'$|J_{tot}|$')
axes[0, 0].set_aspect('equal')
axes[0, 0].set_title(f"1. Poloidal Cut (r, θ)\nz=0")

# Panel 2
mesh2 = axes[0, 1].pcolormesh(Z_grid2, R_grid2, M2, cmap='turbo', vmin=0, vmax=vmax2, shading='gouraud')
plt.colorbar(mesh2, ax=axes[0, 1], fraction=0.046, pad=0.04, label=r'$|J_{tot}|$')
axes[0, 1].set_xlabel("Axial Position (z)")
axes[0, 1].set_ylabel("Radius (r)")
axes[0, 1].set_title("2. Axial Cut (z, r)\nθ=0")

# Panel 3
mesh3 = axes[1, 0].pcolormesh(Z_grid3, T_grid3, M3, cmap='turbo', vmin=0, vmax=vmax3, shading='gouraud')
plt.colorbar(mesh3, ax=axes[1, 0], fraction=0.046, pad=0.04, label=r'$|J_{tot}|$')
axes[1, 0].set_xlabel("Axial Position (z)")
axes[1, 0].set_ylabel("Poloidal Angle (θ)")
axes[1, 0].set_title(f"3. Unrolled q-Surface\nr={r2[r_idx]:.2f}")

# Panel 4
mesh4 = axes[1, 1].pcolormesh(Time_grid4.T, R_grid4.T, Mag4.T, cmap='turbo', vmin=0, vmax=vmax4, shading='gouraud')
plt.colorbar(mesh4, ax=axes[1, 1], fraction=0.046, pad=0.04, label=r'$|J_{tot}|$')
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Radius (r)")
axes[1, 1].set_title("4. Radial Time Evolution\nθ=0, z=0")
time_cursor = axes[1, 1].axvline(t_array[frame_indices[0]], color='white', linewidth=2, linestyle='--')

suptitle = fig.suptitle(f"Time: {t_array[frame_indices[0]]:.2f}", fontsize=16)

def update(frame):
    print(f" -> Processing Frame Index: {frame}")
    M1_new, M2_new, M3_new = calculate_frame_slices(frame)
    
    mesh1.set_array(M1_new.ravel())
    mesh2.set_array(M2_new.ravel())
    mesh3.set_array(M3_new.ravel())
    
    current_time = t_array[frame]
    time_cursor.set_xdata([current_time, current_time])
    suptitle.set_text(f"True Absolute Total Current Evolution | Time: {current_time:.2f} (Idx: {frame})")
    
    return mesh1, mesh2, mesh3, time_cursor, suptitle

ani = FuncAnimation(fig, update, frames=frame_indices, blit=False)

output_file = "C:/Users/mobin/Desktop/tokamak_4D_ABSOLUTE_total.gif"
print(f"\nSaving animation to {output_file}...")
ani.save(output_file, writer='pillow', fps=10)
print("\nSUCCESS! Check your Desktop for the master 4D GIF with Colorbars!")