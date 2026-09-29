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
    j = j + n - nanf[m] + 1
    return int(j)

# ==========================================
# 1. Load Data
# ==========================================
print("Loading massive data arrays...")
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
z_position = 0.0
tolerance = 1e-10

# Animation Time Window (Focusing on the crash phase)
t_start = 650
t_end = 750
step = 2 # Skip every other frame to render faster
frame_indices = range(t_start, t_end, step)

# ==========================================
# 2. Setup Grids
# ==========================================
theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta, indexing='ij') 
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# ==========================================
# 3. The Core Calculation Function
# ==========================================
def compute_total_J_Mag(t_idx):
    """Calculates the 3D vector components of ALL waves and returns total magnitude"""
    Jr_Total = np.zeros_like(R_grid, dtype=float)
    Jt_Total = np.zeros_like(R_grid, dtype=float)
    Jz_Total = np.zeros_like(R_grid, dtype=float)
    
    for i in range(len(mm_array)):
        m = int(mm_array[i])
        n_start = int(nanf_array[i])
        
        for n_step in range(int(nz_array[i])):
            n = n_start + n_step
            mode_idx = janz(m, n, input_folder)
            
            if mode_idx >= bz.shape[2]: continue
            
            br_raw = br[t_idx, :, mode_idx]      
            bt_complex = bt[t_idx, :, mode_idx]  
            bz_complex = bz[t_idx, :, mode_idx]
            
            # Interpolation
            br_complex = np.interp(r2, r1, br_raw.real) + 1j * np.interp(r2, r1, br_raw.imag)
            
            # Gradients
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
            
            # 3D Ampere's Law Math
            k = n / R_major
            jr_1D_complex = 1j * m * bz_over_r - 1j * k * bt_complex
            jt_1D_complex = 1j * k * br_complex - dbz_dr
            jz_1D_complex = bt_over_r + dbt_dr - 1j * m * br_over_r
            
            # Add to respective totals
            if m == 0 and n == 0:
                Jr_Total += jr_1D_complex.real[:, np.newaxis]
                Jt_Total += jt_1D_complex.real[:, np.newaxis]
                Jz_Total += jz_1D_complex.real[:, np.newaxis]
            else:
                phase = m * Theta_grid + k * z_position
                Jr_Total += (jr_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                             jr_1D_complex.imag[:, np.newaxis] * np.sin(phase))
                Jt_Total += (jt_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                             jt_1D_complex.imag[:, np.newaxis] * np.sin(phase))
                Jz_Total += (jz_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                             jz_1D_complex.imag[:, np.newaxis] * np.sin(phase))
                
    # Return the true Pythagorean magnitude
    return np.sqrt(Jr_Total**2 + Jt_Total**2 + Jz_Total**2)

# ==========================================
# 4. Animation Setup
# ==========================================
print(f"Preparing to animate {len(frame_indices)} frames...")

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("X (r * cos θ)")
ax.set_ylabel("Y (r * sin θ)")

print("Calculating initial frame...")
J_Mag_initial = compute_total_J_Mag(frame_indices[0])

# Using 'inferno' because magnitude is strictly positive (no negative values)
mesh = ax.pcolormesh(X, Y, J_Mag_initial, cmap='inferno', shading='gouraud')
cbar = plt.colorbar(mesh, ax=ax, label=r'Total Current Magnitude ($|J_{tot}|$)')
title = ax.set_title("")

def update(frame_idx):
    print(f" -> Rendering Frame: Time Index {frame_idx}...")
    J_Mag_current = compute_total_J_Mag(frame_idx)
    
    mesh.set_array(J_Mag_current.ravel())
    mesh.set_clim(vmin=np.min(J_Mag_current), vmax=np.max(J_Mag_current))
    title.set_text(f"Evolution of $|J_{{tot}}|$\nTime: {t_array[frame_idx]:.2f} (Index: {frame_idx})")
    
    return mesh, title

ani = FuncAnimation(fig, update, frames=frame_indices, blit=False)

# SAVING DIRECTLY TO DESKTOP
# *Change 'mobin' if your Windows username is slightly different!*
output_file = "C:/Users/mobin/Desktop/J_total_magnitude_evolution.gif"

print(f"\nSaving animation to {output_file}...")
ani.save(output_file, writer='pillow', fps=10)

print("\nSUCCESS! Minimize VS Code and check your Desktop for the GIF!")

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
    j = j + n - nanf[m] + 1
    return int(j)

# ==========================================
# 1. Load Data
# ==========================================
print("Loading massive data arrays...")
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
z_position = 0.0
tolerance = 1e-10

# Animation Time Window (Focusing on the crash phase)
t_start = 650
t_end = 750
step = 2 
frame_indices = range(t_start, t_end, step)

# ==========================================
# 2. Setup Grids
# ==========================================
theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta, indexing='ij') 
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# ==========================================
# 3. The Core Calculation Function (PERTURBED ONLY)
# ==========================================
def compute_perturbed_J_Mag(t_idx):
    """Calculates the 3D magnitude EXCLUDING the (0,0) background"""
    Jr_Total = np.zeros_like(R_grid, dtype=float)
    Jt_Total = np.zeros_like(R_grid, dtype=float)
    Jz_Total = np.zeros_like(R_grid, dtype=float)
    
    for i in range(len(mm_array)):
        m = int(mm_array[i])
        n_start = int(nanf_array[i])
        
        for n_step in range(int(nz_array[i])):
            n = n_start + n_step
            
            # PHYSICS FIX: Completely ignore the background to isolate the current sheet!
            if m == 0 and n == 0:
                continue
                
            mode_idx = janz(m, n, input_folder)
            if mode_idx >= bz.shape[2]: continue
            
            br_raw = br[t_idx, :, mode_idx]      
            bt_complex = bt[t_idx, :, mode_idx]  
            bz_complex = bz[t_idx, :, mode_idx]
            
            # Interpolation
            br_complex = np.interp(r2, r1, br_raw.real) + 1j * np.interp(r2, r1, br_raw.imag)
            
            # Gradients
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
            
            # 3D Ampere's Law Math
            k = n / R_major
            jr_1D_complex = 1j * m * bz_over_r - 1j * k * bt_complex
            jt_1D_complex = 1j * k * br_complex - dbz_dr
            jz_1D_complex = bt_over_r + dbt_dr - 1j * m * br_over_r
            
            # Phase rotation
            phase = m * Theta_grid + k * z_position
            Jr_Total += (jr_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                         jr_1D_complex.imag[:, np.newaxis] * np.sin(phase))
            Jt_Total += (jt_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                         jt_1D_complex.imag[:, np.newaxis] * np.sin(phase))
            Jz_Total += (jz_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                         jz_1D_complex.imag[:, np.newaxis] * np.sin(phase))
                
    return np.sqrt(Jr_Total**2 + Jt_Total**2 + Jz_Total**2)

# ==========================================
# 4. Animation Setup with FIXED COLORS
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("X (r * cos θ)")
ax.set_ylabel("Y (r * sin θ)")

# To fix the color scale, we calculate a "Peak Crash" frame first to find a good global maximum
print("Pre-scanning the peak crash frame to lock the color scale...")
J_Mag_peak = compute_perturbed_J_Mag(710) # 710 is roughly right around the crash
global_vmax = np.max(J_Mag_peak) * 1.05   # Add 5% headroom so the sheet doesn't wash out white

print("Calculating initial frame...")
J_Mag_initial = compute_perturbed_J_Mag(frame_indices[0])

# We set vmin=0 and vmax=global_vmax so the "camera" never changes!
mesh = ax.pcolormesh(X, Y, J_Mag_initial, cmap='turbo', vmin=0, vmax=global_vmax, shading='gouraud')
cbar = plt.colorbar(mesh, ax=ax, label=r'Perturbed Current Magnitude ($|\delta J_{tot}|$)')
title = ax.set_title("")

def update(frame_idx):
    print(f" -> Rendering Frame: Time Index {frame_idx}...")
    J_Mag_current = compute_perturbed_J_Mag(frame_idx)
    
    mesh.set_array(J_Mag_current.ravel())
    # Notice we REMOVED mesh.set_clim() here so the colors stay locked!
    
    title.set_text(f"Isolated Current Sheet Evolution\nTime: {t_array[frame_idx]:.2f} (Index: {frame_idx})")
    return mesh, title

ani = FuncAnimation(fig, update, frames=frame_indices, blit=False)

output_file = "C:/Users/mobin/Desktop/current_sheet_isolated.gif"
print(f"\nSaving animation to {output_file}...")
ani.save(output_file, writer='pillow', fps=10)
print("\nSUCCESS! Check your Desktop for the GIF!")