'''
This script demonstrates how to use c2raytools to combine several
coeval data volumes into a lightcone volume. 
'''

import c2raytools as c2t
import numpy as np
import pylab as pl
import glob

#Directories to look for files in. Modify as needed to point to directories
#that are available on your system.
base_path = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/' 
xfrac_dir = base_path+'/114Mpc_f2_10S_256/results_ranger/'
density_dir = base_path+'/coarser_densities/nc256_halos_included/'

#Enable output
c2t.set_verbose(True)

#We are using the 114/h Mpc simulation box, so set all the proper conversion factors
#Always set this before loading data!
c2t.conv.set_sim_constants(boxsize_cMpc = 114.)

#We will make one density light cone, one for ionized
#fraction and one for brightness temperature.

#First we need lists of all the available ionization fraction and density files.
#Here, we use python's built-in glob command, which makes a list of all such files:
xfrac_files = glob.glob(xfrac_dir + '/xfrac3d_*.bin')
density_files = glob.glob(density_dir + '/*n_all.dat')

#Make the ionization fraction lightcone
xfrac_lightcone, z = c2t.make_lightcone(xfrac_files, z_low = 7.059, z_high = 7.3)

#Make the density lightcone
density_lightcone, z = c2t.make_lightcone(density_files, z_low = 7.059, z_high = 7.3)

#The dT lightcone is made by combining density and ionization fraction.
#If you already have a bunch of dT files, you can make a lightcone directly
#using the make_lightcone method
dt_lightcone = c2t.calc_dt_lightcone(xfrac_lightcone, density_lightcone, lowest_z = z.min())
print dt_lightcone.shape

#Plot the dT volume
pl.imshow(dt_lightcone[0,:,:], extent=[z.min(),z.max(), 0, c2t.conv.LB], aspect='auto')
pl.colorbar()
pl.xlabel('$z$')
pl.ylabel('Mpc')
pl.show()

pl.show()


