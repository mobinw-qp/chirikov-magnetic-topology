import numpy as np
import plotly.graph_objects as go
from scipy.io import readsav

# --- 1. Load Data ---
input_folder = "C:/data.specly/"
fields = readsav(input_folder + 'modeprofiles_10000.sav')
br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))
r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

def janz(m, n, directory):
    spectr = readsav(directory + 'spectrum.sav')
    nanf, nz = spectr['NANF'].view(), spectr['NZ'].view()
    j = 0
    if m == 0 and n == 0: return int(j)
    for im in range(m): j += nz[im]
    return int(j + n - nanf[m] + 1)

R_major = 4.0
t_idx = 710 
m, n = 1, -1
mode_idx = janz(m, n, input_folder)

# --- 2. Physics Math (Calculated in 1D first) ---
br_raw = br[t_idx, :, mode_idx]
bt_full = bt[t_idx, :, mode_idx]
bz_full = bz[t_idx, :, mode_idx]

br_complex = np.interp(r2, r1, br_raw.real) + 1j * np.interp(r2, r1, br_raw.imag)

dbt_dr = np.gradient(bt_full, r2)
dbz_dr = np.gradient(bz_full, r2)

tolerance = 1e-10
safe_mask = r2 > tolerance
bt_over_r = np.zeros_like(bt_full)
br_over_r = np.zeros_like(br_complex)
bz_over_r = np.zeros_like(bz_full)

bt_over_r[safe_mask] = bt_full[safe_mask] / r2[safe_mask]
br_over_r[safe_mask] = br_complex[safe_mask] / r2[safe_mask]
bz_over_r[safe_mask] = bz_full[safe_mask] / r2[safe_mask]

bt_over_r[~safe_mask] = dbt_dr[~safe_mask]
br_over_r[~safe_mask] = np.gradient(br_complex, r2)[~safe_mask]
bz_over_r[~safe_mask] = dbz_dr[~safe_mask]

k = n / R_major
jr_1D = 1j * m * bz_over_r - 1j * k * bt_full
jt_1D = 1j * k * br_complex - dbz_dr
jz_1D = bt_over_r + dbt_dr - 1j * m * br_over_r

# --- 3. Build a TRUE Cartesian 3D Grid ---
# Plotly strictly requires a rectangular box (X, Y, Z) to render isosurfaces!
grid_res = 60 # 60x60x60 grid is a great balance of detail and browser speed
x_1d = np.linspace(-1, 1, grid_res)
y_1d = np.linspace(-1, 1, grid_res)
z_1d = np.linspace(0, 2 * np.pi * R_major, grid_res)

X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

# Convert the X,Y coordinates of our box into Radius and Theta
R_grid = np.sqrt(X**2 + Y**2)
T_grid = np.arctan2(Y, X)

# --- 4. Map Physics to the 3D Grid ---
# Interpolate the 1D physics arrays onto the 3D radius grid
jr_3d_real = np.interp(R_grid.flatten(), r2, jr_1D.real).reshape(X.shape)
jr_3d_imag = np.interp(R_grid.flatten(), r2, jr_1D.imag).reshape(X.shape)
jt_3d_real = np.interp(R_grid.flatten(), r2, jt_1D.real).reshape(X.shape)
jt_3d_imag = np.interp(R_grid.flatten(), r2, jt_1D.imag).reshape(X.shape)
jz_3d_real = np.interp(R_grid.flatten(), r2, jz_1D.real).reshape(X.shape)
jz_3d_imag = np.interp(R_grid.flatten(), r2, jz_1D.imag).reshape(X.shape)

# Apply Phase Rotation in full 3D space
phase = m * T_grid + k * Z
Jr_3D = jr_3d_real * np.cos(phase) - jr_3d_imag * np.sin(phase)
Jt_3D = jt_3d_real * np.cos(phase) - jt_3d_imag * np.sin(phase)
Jz_3D = jz_3d_real * np.cos(phase) - jz_3d_imag * np.sin(phase)

J_Mag_3D = np.sqrt(Jr_3D**2 + Jt_3D**2 + Jz_3D**2)

# Mask out the corners of the rectangular box that fall outside our cylindrical plasma (r > 1)
J_Mag_3D[R_grid > 1.0] = 0.0

# --- 5. Plotly 3D Isosurface ---
j_max = np.max(J_Mag_3D)

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=J_Mag_3D.flatten(),
    isomin=j_max * 0.65, # Threshold: Only show the top 35% of the most intense current
    isomax=j_max,
    surface_count=5, # 5 nested layers of intensity
    colorscale='inferno',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

fig.update_layout(
    title="3D Localized Current Sheet (m=1, n=-1 Tearing Mode)", 
    scene=dict(
        aspectmode='data',
        xaxis_title='X', yaxis_title='Y', zaxis_title='Axial Length (Z)',
        # Hides the ugly background grid walls for a cleaner look
        xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)
    )
)

output_file = "C:/Users/mobin/Desktop/3D_Current_Sheet.html"
fig.write_html(output_file)
print(f"SUCCESS! Saved 3D interactive model to {output_file}")