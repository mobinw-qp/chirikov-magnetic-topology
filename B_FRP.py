import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

# Updated to point to the new RFP data folder
input_folder = "C:/data.specly/RFP/"

def janz(m, n, directory):
    """Return the value of the array corresponding to the specified mode (m,n)"""
    if not directory.endswith('/'):
        directory += '/'
        
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

print("Loading RFP data...")
fields = readsav(input_folder + 'modeprofiles.sav')

# UPDATED KEYS: Using the exact names from dict_keys
t = fields['t'].view()

br = np.transpose(fields['br'].view(), (2, 1, 0))
bt = np.transpose(fields['bt'].view(), (2, 1, 0))
bz = np.transpose(fields['bz'].view(), (2, 1, 0))
vr = np.transpose(fields['vr'].view(), (2, 1, 0))
vt = np.transpose(fields['vt'].view(), (2, 1, 0))
vz = np.transpose(fields['vz'].view(), (2, 1, 0))

r1 = fields['pror1'].view()
r2 = fields['pror2'].view()

# Extract the index for the equilibrium background mode
mode = janz(0, 0, input_folder)

# We will plot the final time step (-1) instead of the first (0) to ensure 
# the RFP equilibrium is fully formed in the simulation
t_idx = -1 

plt.figure(figsize=(8, 5))
plt.plot(r2, bz[t_idx, :, mode].real, color='blue', label=r"Axial Field ($B_z$)", linestyle='-', linewidth=2)
plt.plot(r2, bt[t_idx, :, mode].real, color='red', label=r"Poloidal Field ($B_{\theta}$)", linestyle='-', linewidth=2)

# Add a horizontal line at 0 to easily spot the RFP reversal
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

# FIXED: Re-inserted the \n character for the line break
plt.title(f"RFP Equilibrium Magnetic Fields (m=0, n=0)\nTime = {t[t_idx]:.2f}")
plt.xlabel("Radius (r)")
plt.ylabel("Magnetic Field Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 1)

plt.tight_layout()
plt.show()