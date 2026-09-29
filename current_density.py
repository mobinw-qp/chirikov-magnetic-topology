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
# NEW CODE: Calculating Current Densities
# Assuming mu_0 = 1, so J = curl(B)
# ==========================================

# 1. Extract the 1D arrays for the m=0, n=0 mode for cleaner math
bz_00 = bz[0, :, mode]
bt_00 = bt[0, :, mode]

# 2. Calculate J_theta = - d(B_z)/dr
j_theta = -np.gradient(bz_00, r2)

# 3. Calculate J_z = (B_theta / r) + d(B_theta)/dr
# Calculate the derivative of B_theta directly
dbt_dr = np.gradient(bt_00, r2)

# Initialize the division array
b_theta_over_r = np.zeros(r2.shape, dtype=complex)

# Safely divide B_theta by r using a small tolerance to catch floating-point 'zeros'
tolerance = 1e-10
safe_mask = r2 > tolerance
np.divide(bt_00, r2, out=b_theta_over_r, where=safe_mask)

# Apply L'Hopital's rule for the origin limit: lim(r->0) [B_theta / r] = d(B_theta)/dr
zero_mask = r2 <= tolerance
b_theta_over_r[zero_mask] = dbt_dr[zero_mask]

# J_z is the sum of the two terms
j_z = b_theta_over_r + dbt_dr

# 4. Plot the Current Densities
plt.figure(figsize=(8, 6))

plt.plot(r2, j_z.real, color='purple', label=r"$J_z$", linestyle='-')
plt.plot(r2, j_theta.real, color='orange', label=r"$J_{\theta}$", linestyle='-')

plt.axhline(0, color='black', linewidth=1, linestyle='--')

plt.xlabel('Radius (r)')
plt.ylabel('Current Density (J)')
plt.title('Equilibrium Current Profiles (m=0, n=0)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# ==========================================
# NEW CODE: Parallel and Perpendicular Components
# ==========================================

# 1. Extract real parts for physical calculations
bz_real = bz_00.real
bt_real = bt_00.real
jz_real = j_z.real
jt_real = j_theta.real

# 2. Calculate B squared: B^2 = B_z^2 + B_theta^2
B_sq = bz_real**2 + bt_real**2

# 3. Calculate Dot product: J . B = J_z*B_z + J_theta*B_theta
J_dot_B = jz_real * bz_real + jt_real * bt_real

# 4. Scalar projection factor: alpha = (J . B) / B^2
# Safe division in case B_sq is zero anywhere (though unlikely in a plasma core)
alpha = np.zeros_like(B_sq)
np.divide(J_dot_B, B_sq, out=alpha, where=(B_sq != 0))

# 5. Parallel components (J_par = alpha * B)
# j_par_r = 0 (omitted since it's zero everywhere)
j_par_t = alpha * bt_real
j_par_z = alpha * bz_real

# 6. Perpendicular components (J_perp = J - J_par)
# j_perp_r = 0 (omitted)
j_perp_t = jt_real - j_par_t
j_perp_z = jz_real - j_par_z

# ==========================================
# FIX: Mask out the r=0 point for plotting
# ==========================================
# Create a boolean mask where True means we keep the data (r2 is not exactly 0)
mask = (r2 != 0)

# Apply the mask to all our plotting arrays
r2_plot = r2[mask]

jz_plot = jz_real[mask]
j_par_z_plot = j_par_z[mask]
j_perp_z_plot = j_perp_z[mask]

jt_plot = jt_real[mask]
j_par_t_plot = j_par_t[mask]
j_perp_t_plot = j_perp_t[mask]

# ==========================================
# Plotting
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Z components
ax1.plot(r2_plot, jz_plot, color='purple', label=r"$J_z$ (Total)", linestyle='-')
ax1.plot(r2_plot, j_par_z_plot, color='blue', label=r"$J_{\parallel, z}$", linestyle='--')
ax1.plot(r2_plot, j_perp_z_plot, color='red', label=r"$J_{\perp, z}$", linestyle='-.')
ax1.axhline(0, color='black', linewidth=1, linestyle=':')
ax1.set_xlabel('Radius (r)')
ax1.set_ylabel('Current Density')
ax1.set_title('Axial Current Density Components (m=0, n=0)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Theta components
ax2.plot(r2_plot, jt_plot, color='orange', label=r"$J_{\theta}$ (Total)", linestyle='-')
ax2.plot(r2_plot, j_par_t_plot, color='blue', label=r"$J_{\parallel, \theta}$", linestyle='--')
ax2.plot(r2_plot, j_perp_t_plot, color='red', label=r"$J_{\perp, \theta}$", linestyle='-.')
ax2.axhline(0, color='black', linewidth=1, linestyle=':')
ax2.set_xlabel('Radius (r)')
ax2.set_ylabel('Current Density')
ax2.set_title('Poloidal Current Density Components (m=0, n=0)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()