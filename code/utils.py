import numpy as np

# from astropy.cosmology import FlatLambdaCDM
### Cosmology of the simulation we use, SDSS absolute magnitude
### measurement effectively independent of cosmology
# cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307115) # smdpl, the same as Planck13 of astropy.cosmology
### cosmology,changed --wenlk, 24/07/08
### from astropy.cosmology import Planck13
from astropy.cosmology import Planck13  # Planck (2014) Paper XVI, Table 5
cosmo = Planck13

from scipy.interpolate import interp1d
# Conversion between redshift and comoving distance in Mpc (not Mpc/h)
z_interp = np.linspace(0.0, 1.0, 100000)
comoving_distance_interp = cosmo.comoving_distance(z_interp).value
get_comoving_distance = interp1d(z_interp, comoving_distance_interp)
get_z = interp1d(comoving_distance_interp, z_interp)
# Conversion between redshift and comoving distance in Mpc/h (not Mpc)
comoving_distance_interp_h = comoving_distance_interp * cosmo.h
get_comoving_distance_h = interp1d(z_interp, comoving_distance_interp_h)
get_z_h = interp1d(comoving_distance_interp_h, z_interp)

from scipy import constants
# Speed of light in km/s
c = constants.c/1e3

# Sample selection criteria from Zehavi 2011
z_min = 0.02
M_thr_arr = np.array((-21.0, -20.5, -20.0, -19.5, -19.0, -20.0, -19.5))
z_max_arr = np.array((47650, 39700, 31900, 25450, 19250, 19250, 19250)) / c
thr_list  = ['21p0', '20p5', '20p0', '19p5', '19p0', '20p0x', '19p5x']

ngal_arr  = np.array((1.128, 3.044, 6.192, 11.32, 15.28, 5.985, 10.19)) * 0.001# Fig 3 of Wang et al. 2022

# comoving volume in units of (Mpc/h)^3
### angarea,changed --wenlk, 24/07/08
# def Vcom(zmax, angarea = 7461):
def Vcom(zmax, angarea = 7320):
    return (4 * np.pi / 3 * (get_comoving_distance(zmax) ** 3 - get_comoving_distance(0.02) ** 3)
            * cosmo.h ** 3 * angarea / 41253)

#######################################################################################
### 2024/06/14, import pymangle error, for fiber collision/completeness, no need to use
### 2024/07/08, files with '_pix' are calculated by mangle2.2 --- File_description.md
### 2024/07/08, file changed with bright1
#######################################################################################
# import pymangle
# from astropy.io import fits
# window   = pymangle.Mangle('window_pix.dr72bright1.ply')
# mask     = pymangle.Mangle('mask_pix.dr72bright1.ply')
# combmask = pymangle.Mangle('lss_combmask_pix.dr72.ply')
# combmask_info  = fits.open('lss_combmask.dr72.fits')[1].data
# mangle   = [window, mask, combmask, combmask_info]
###############################################################################


from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['mathtext.rm'] = 'serif'
rcParams['mathtext.it'] = 'serif:italic'
rcParams['mathtext.bf'] = 'serif:bold'
rcParams['axes.titlepad'] = 12
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['figure.dpi'] = 80
rcParams['figure.figsize'] = [8, 6]
rcParams['figure.titlesize'] = 25
rcParams['font.size'] = 20.0
rcParams['legend.fontsize'] = 20
rcParams['legend.frameon'] = False
rcParams['savefig.pad_inches'] = 0.1
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
