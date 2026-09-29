import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

# --- 1. Load Data ---
input_folder = "C:/data.specly/"
fields = readsav(input_folder + 'modeprofiles_10000.sav')
br = np.transpose(fields['prodbr'].view(), (2, 1, 0))
bt = np.transpose(fields['prodbt'].view(), (2, 1, 0))
bz = np.transpose(fields['prodbz'].view(), (2, 1, 0))
r1 = fields['pror1'].view() 
r2 = fields['pror2'].view() 
t_array = fields['prot'].view()

def janz(m, n, directory):
    spectr = readsav(directory + 'spectrum.sav')
    nanf, nz = spectr['NANF'].view(), spectr['NZ'].view()
    j = 0
    if m == 0 and n == 0: return int(j)
    for im in range(m): j += nz[im]
    return int(j + n - nanf[m] + 1)

R_major = 4.0
tolerance = 1e-10
t_idx = 710 
m, n = 1, -1

# --- 2. Physics Math ---
mode_idx = janz(m, n, input_folder)

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

bt_over_r[safe_mask] = bt_complex[safe_mask] / r2[safe_mask]
br_over_r[safe_mask] = br_complex[safe_mask] / r2[safe_mask]
bz_over_r[safe_mask] = bz_complex[safe_mask] / r2[safe_mask]

bt_over_r[~safe_mask] = dbt_dr[~safe_mask]
br_over_r[~safe_mask] = np.gradient(br_complex, r2)[~safe_mask]
bz_over_r[~safe_mask] = dbz_dr[~safe_mask]

k = n / R_major
jr_1D = 1j * m * bz_over_r - 1j * k * bt_complex
jt_1D = 1j * k * br_complex - dbz_dr
jz_1D = bt_over_r + dbt_dr - 1j * m * br_over_r

# --- 3. 2D Mapping ---
theta = np.linspace(0, 2 * np.pi, 200)
R_grid, Theta_grid = np.meshgrid(r2, theta, indexing='ij') 
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

phase = m * Theta_grid + k * 0.0 # z = 0
Jr = jr_1D.real[:, np.newaxis] * np.cos(phase) - jr_1D.imag[:, np.newaxis] * np.sin(phase)
Jt = jt_1D.real[:, np.newaxis] * np.cos(phase) - jt_1D.imag[:, np.newaxis] * np.sin(phase)
Jz = jz_1D.real[:, np.newaxis] * np.cos(phase) - jz_1D.imag[:, np.newaxis] * np.sin(phase)

J_Mag = np.sqrt(Jr**2 + Jt**2 + Jz**2)

# --- 4. Thresholded Plotting ---
plt.figure(figsize=(8, 7))

# Set background to black
plt.gca().set_facecolor('black')

# Calculate the threshold: Only show values greater than 60% of the maximum current
j_max = np.max(J_Mag)
threshold_min = j_max * 0.60 

# Create custom levels that start AT the threshold
levels = np.linspace(threshold_min, j_max, 50)

# The contourf will only draw shapes for data inside these levels.
cont = plt.contourf(X, Y, J_Mag, levels=levels, cmap='inferno', extend='max')

# Add the q=1 surface reference ring
q_radius = 0.42
circle = plt.Circle((0, 0), q_radius, color='white', fill=False, linestyle=':', alpha=0.5, label='q=1 surface')
plt.gca().add_patch(circle)

plt.colorbar(cont, label=r'Isolated Current Sheet Magnitude ($|\delta J_{tot}|$)')
plt.title(f"2D Thresholded Current Sheet\nTime: {t_array[t_idx]:.0f} | Mode: ({m}, {n})")
plt.xlabel("X (r * cos θ)")
plt.ylabel("Y (r * sin θ)")
plt.xlim(-0.8, 0.8)
plt.ylim(-0.8, 0.8)
plt.legend(loc='upper right')
plt.gca().set_aspect('equal')
plt.show()