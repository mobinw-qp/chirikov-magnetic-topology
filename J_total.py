import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

input_folder = "C:/data.specly/"

def janz(m, n, directory):
    spectr_arrays = readsav(directory + 'spectrum.sav')
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
# 1. Load Data
# ==========================================
print("Loading data...")
fields = readsav(input_folder + 'modeprofiles_10000.sav')
t = fields['prot'].view()

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
t_idx = 700 
z_position = 0.0
tolerance = 1e-10

# ==========================================
# 2. Setup Your Custom Selections
# ==========================================
modes_to_isolate = [
    (1, -1), 
    (2, -2)
]

theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta, indexing='ij') 
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# We need separate canvases to accumulate the vectors for the Grand Total
Jr_Total_2D = np.zeros_like(R_grid, dtype=float)
Jt_Total_2D = np.zeros_like(R_grid, dtype=float)
Jz_Total_2D = np.zeros_like(R_grid, dtype=float)

separate_mode_data = {} 

# ==========================================
# 3A. Calculate the TRUE Grand Total Vectors
# ==========================================
print("Calculating Grand Total vectors of all perturbed modes...")
for i in range(len(mm_array)):
    m = int(mm_array[i])
    n_start = int(nanf_array[i])
    for step in range(int(nz_array[i])):
        n = n_start + step
        
        # Skip background so we can see the perturbations clearly
        if m == 0 and n == 0:
            continue
            
        mode_idx = janz(m, n, input_folder)
        if mode_idx >= bz.shape[2]:
            continue
        
        # Extract fields
        br_raw = br[t_idx, :, mode_idx]      
        bt_complex = bt[t_idx, :, mode_idx]  
        bz_complex = bz[t_idx, :, mode_idx]  # Bz is already on the r2 grid!
        
        # Interpolate Br
        br_real_interp = np.interp(r2, r1, br_raw.real)
        br_imag_interp = np.interp(r2, r1, br_raw.imag)
        br_complex = br_real_interp + 1j * br_imag_interp
        
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
        
        # MATHEMATICS: Ampere's Law for all 3 components
        k = n / R_major
        jr_1D_complex = 1j * m * bz_over_r - 1j * k * bt_complex
        jt_1D_complex = 1j * k * br_complex - dbz_dr
        jz_1D_complex = bt_over_r + dbt_dr - 1j * m * br_over_r
        
        # Phase rotation
        phase = m * Theta_grid + k * z_position
        
        jr_2D_mode = (jr_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                      jr_1D_complex.imag[:, np.newaxis] * np.sin(phase))
        jt_2D_mode = (jt_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                      jt_1D_complex.imag[:, np.newaxis] * np.sin(phase))
        jz_2D_mode = (jz_1D_complex.real[:, np.newaxis] * np.cos(phase) - 
                      jz_1D_complex.imag[:, np.newaxis] * np.sin(phase))
        
        # Accumulate Vector Components
        Jr_Total_2D += jr_2D_mode
        Jt_Total_2D += jt_2D_mode
        Jz_Total_2D += jz_2D_mode

# Calculate Grand Total Magnitude
J_Mag_Grand_Total = np.sqrt(Jr_Total_2D**2 + Jt_Total_2D**2 + Jz_Total_2D**2)

# ==========================================
# 3B. Force-Calculate Your Isolated Modes
# ==========================================
print("\nExtracting specific isolated modes...")
for (m, n) in modes_to_isolate:
    mode_idx = janz(m, n, input_folder)
    print(f" -> Forced isolation for m={m}, n={n} (Extracted Index: {mode_idx})")
    
    if mode_idx >= bz.shape[2]:
        continue
        
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
    jr_1D_complex = 1j * m * bz_over_r - 1j * k * bt_complex
    jt_1D_complex = 1j * k * br_complex - dbz_dr
    jz_1D_complex = bt_over_r + dbt_dr - 1j * m * br_over_r
    
    phase = m * Theta_grid + k * z_position
    
    jr_2D = jr_1D_complex.real[:, np.newaxis] * np.cos(phase) - jr_1D_complex.imag[:, np.newaxis] * np.sin(phase)
    jt_2D = jt_1D_complex.real[:, np.newaxis] * np.cos(phase) - jt_1D_complex.imag[:, np.newaxis] * np.sin(phase)
    jz_2D = jz_1D_complex.real[:, np.newaxis] * np.cos(phase) - jz_1D_complex.imag[:, np.newaxis] * np.sin(phase)
    
    # Calculate Magnitude for just this specific mode
    j_mag_mode = np.sqrt(jr_2D**2 + jt_2D**2 + jz_2D**2)
    separate_mode_data[(m, n)] = j_mag_mode

# ==========================================
# 4. Multi-Panel Plotting
# ==========================================
num_plots = len(modes_to_isolate) + 1
fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

# We use 'inferno' because magnitude is strictly positive
for i, (m, n) in enumerate(modes_to_isolate):
    ax = axes[i]
    if (m, n) in separate_mode_data:
        cont = ax.contourf(X, Y, separate_mode_data[(m, n)], levels=60, cmap='inferno')
        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Isolated $|J_{{tot}}|$\nm={m}, n={n}")
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

ax_total = axes[-1]
cont_total = ax_total.contourf(X, Y, J_Mag_Grand_Total, levels=60, cmap='inferno')
plt.colorbar(cont_total, ax=ax_total, fraction=0.046, pad=0.04, label=r'Current Magnitude ($|J_{tot}|$)')
ax_total.set_title("Grand Total Perturbed Magnitude\n$|J_{tot}| = \sqrt{J_r^2 + J_\\theta^2 + J_z^2}$")
ax_total.set_aspect('equal')
ax_total.set_xlim(-1, 1)
ax_total.set_ylim(-1, 1)

plt.tight_layout()
plt.show()
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
# 3. The Core Calculation Function (TOTAL J)
# ==========================================
# FIX: Manually build the list of modes and INJECT the (0,0) background first!
all_modes_with_background = [(0, 0)]
for i in range(len(mm_array)):
    m_val = int(mm_array[i])
    n_start = int(nanf_array[i])
    for n_step in range(int(nz_array[i])):
        n_val = n_start + n_step
        if m_val == 0 and n_val == 0: continue # Prevent duplicates
        all_modes_with_background.append((m_val, n_val))

def compute_total_J_Mag(t_idx):
    """Calculates the 3D magnitude EXPLICITLY INCLUDING the (0,0) background"""
    Jr_Total = np.zeros_like(R_grid, dtype=float)
    Jt_Total = np.zeros_like(R_grid, dtype=float)
    Jz_Total = np.zeros_like(R_grid, dtype=float)
    
    # Notice we now loop over our custom list instead of mm_array!
    for (m, n) in all_modes_with_background:
        
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
        
        # Phase rotation (background (0,0) is symmetric, no rotation needed)
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
J_Mag_peak = compute_total_J_Mag(710) # 710 is roughly right around the crash
global_vmax = np.max(J_Mag_peak) * 1.05   # Add 5% headroom

print("Calculating initial frame...")
J_Mag_initial = compute_total_J_Mag(frame_indices[0])

# We set vmin=0 and vmax=global_vmax so the "camera" never changes!
mesh = ax.pcolormesh(X, Y, J_Mag_initial, cmap='turbo', vmin=0, vmax=global_vmax, shading='gouraud')
cbar = plt.colorbar(mesh, ax=ax, label=r'Total Current Magnitude ($|J_{tot}|$)')
title = ax.set_title("")

def update(frame_idx):
    print(f" -> Rendering Frame: Time Index {frame_idx}...")
    J_Mag_current = compute_total_J_Mag(frame_idx)
    
    mesh.set_array(J_Mag_current.ravel())
    # Notice we REMOVED mesh.set_clim() here so the colors stay locked!
    
    title.set_text(f"Total Current Evolution (Background Included)\nTime: {t_array[frame_idx]:.2f} (Index: {frame_idx})")
    return mesh, title

ani = FuncAnimation(fig, update, frames=frame_indices, blit=False)

output_file = "C:/Users/mobin/Desktop/current_sheet_total.gif"
print(f"\nSaving animation to {output_file}...")
ani.save(output_file, writer='pillow', fps=10)

import numpy as np
import plotly.graph_objects as go
from scipy.io import readsav

# --- 1. Load Data (Simplified setup for 1 mode) ---
input_folder = "C:/data.specly/"
fields = readsav(input_folder + 'modeprofiles_10000.sav')
br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))
r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

R_major = 4.0
t_idx = 710 
mode_idx = 3 # Manually replace with your exact (1, -1) array index from janz!
m, n = 1, -1

# --- 2. Build Full 3D Grid ---
r_sub = r2[::2] # Downsample slightly for 3D rendering speed
theta = np.linspace(0, 2*np.pi, 40)
z = np.linspace(0, 2*np.pi*R_major, 60)
R, T, Z = np.meshgrid(r_sub, theta, z, indexing='ij')

X = R * np.cos(T)
Y = R * np.sin(T)

# --- 3. Compute 3D Perturbed Current ---
br_raw, bt_complex, bz_complex = br[t_idx, ::2, mode_idx], bt[t_idx, ::2, mode_idx], bz[t_idx, ::2, mode_idx]
br_complex = np.interp(r_sub, r1, br_raw.real) + 1j * np.interp(r_sub, r1, br_raw.imag)

dbt_dr = np.gradient(bt_complex, r_sub)
dbz_dr = np.gradient(bz_complex, r_sub)

# (Skipping origin safety checks for brevity - apply your previous L'Hopital masks here)
bt_over_r = bt_complex / (r_sub + 1e-10)
br_over_r = br_complex / (r_sub + 1e-10)
bz_over_r = bz_complex / (r_sub + 1e-10)

k = n / R_major
jr_1D = 1j * m * bz_over_r - 1j * k * bt_complex
jt_1D = 1j * k * br_complex - dbz_dr
jz_1D = bt_over_r + dbt_dr - 1j * m * br_over_r

phase = m * T + k * Z
Jr = jr_1D.real[:, np.newaxis, np.newaxis] * np.cos(phase) - jr_1D.imag[:, np.newaxis, np.newaxis] * np.sin(phase)
Jt = jt_1D.real[:, np.newaxis, np.newaxis] * np.cos(phase) - jt_1D.imag[:, np.newaxis, np.newaxis] * np.sin(phase)
Jz = jz_1D.real[:, np.newaxis, np.newaxis] * np.cos(phase) - jz_1D.imag[:, np.newaxis, np.newaxis] * np.sin(phase)

J_Mag_3D = np.sqrt(Jr**2 + Jt**2 + Jz**2)

# --- 4. Plotly 3D Isosurface ---
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=J_Mag_3D.flatten(),
    isomin=np.max(J_Mag_3D)*0.6, # Only show the top 40% of intense current!
    isomax=np.max(J_Mag_3D),
    surface_count=5, # Number of layers
    colorscale='Turbo',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

fig.update_layout(title="3D Localized Current Sheet (Tearing Mode)", scene=dict(aspectmode='data'))
fig.write_html("C:/Users/mobin/Desktop/3D_Current_Sheet.html")
print("Saved 3D model to Desktop!")