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
