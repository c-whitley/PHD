import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import binned_statistic

## Physical powerspectrum P(k)
# Make gaussian random field with given power spectrum 
# define number of pixels and size of box
n = 2048 * 2
box_size = 1000 #Mpc

# generate k values and scale by box size
kpix = 2.0 * np.pi / box_size
kx = np.fft.fftfreq(n) * n * kpix
ky = np.fft.fftfreq(n) * n * kpix
k = np.sqrt(kx[None,:]**2 + ky[:,None]**2)

# load in a P(k)
data = np.loadtxt('ics_matterpow_127.dat')
k_in = data[:,0]
P_in = data[:,1]
P_in[k_in>1E-2] = 0

# interpolate input P(k) values over generated k values
P_interp = interpolate.interp1d(k_in, P_in, bounds_error=False, fill_value=0.0)
P = P_interp(k)

# make grid with the same shape as k filled with complex numbers 
# complex number, a, must fullfil condition P = abs(a)**2
# offset each complex part by gaussian with mean=0 and std=1
real_part = np.sqrt(0.5*P) * np.random.normal(loc=0.0,scale=1.0,size=k.shape)
imaginary_part = np.sqrt(0.5*P) * np.random.normal(loc=0.0,scale=1.0,size=k.shape)
ft_map = (real_part + imaginary_part*1.0j)

# inverse FFT the grid of complex numbers to get real map
gauss_field = np.fft.ifft2(ft_map)

# Calculate power spectrum from real map
# FFT the map and calculate the power
n_f = np.fft.fft2(gauss_field)
P_out = np.abs(n_f )**2

# define bins of k values with minimum bin size, delta_k=1/box_size
k_max = np.max(k)
delta_k = 1.0 / box_size * 20
k_out = np.arange(int(np.floor(k_max/delta_k))+1) * delta_k

# bin the power and calulate the mean in each bin
k_flat = np.ndarray.flatten(k)
P_flat = np.ndarray.flatten(P_out)
P_out, _, _ = binned_statistic(k_flat, P_flat, bins = k_out)
k_out = (k_out[:-1] + k_out[1:]) * 0.5

## Plotting
# set up figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.set_size_inches(17,5)
plt.subplots_adjust(wspace=0.3)

# input power spectrum
ax1.plot(k_in, P_in)
ax1.set_ylabel('P(k)')
ax1.set_xlabel('k')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.axvline(kx[1], color='k', linestyle='--', linewidth=0.8)
ax1.axvline(np.max(k), color='k', linestyle='--', linewidth=0.8)

# gaussian random fireld with input P(k) 
im = ax2.imshow(gauss_field.real, extent=[0,box_size,0,box_size], cmap='jet')
ax2.set_xlabel('Mpc')
ax2.set_ylabel('Mpc')
plt.colorbar(im, ax=ax2, shrink=0.75)

# calculated P(k) from gaussian random field and input P(k)
ax3.plot(k_in, P_in, 'r')
ax3.plot(k_out, P_out)
ax3.set_ylabel('P(k)')
ax3.set_xlabel('k')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(ax1.get_xlim())
ax3.set_ylim(ax1.get_ylim())
ax3.axvline(kx[1], color='k', linestyle='--', linewidth=0.8)
ax3.axvline(np.max(k), color='k', linestyle='--', linewidth=0.8)

plt.show()w