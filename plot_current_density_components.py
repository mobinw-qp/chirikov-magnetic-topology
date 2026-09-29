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