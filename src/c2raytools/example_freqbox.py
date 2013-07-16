import c2raytools as c2t
import numpy as np
import pylab as pl

#Directories to look for files in. Modify as needed
base_path = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
xfrac_dir = base_path+'/114Mpc_f2_10S_256/results_ranger/'
density_dir = base_path+'/coarser_densities/nc256_halos_removed/'

#Enable output
c2t.set_verbose(True)

#We are using the 114/h Mpc simulation box, so set all the proper conversion factors
c2t.conv.set_sim_constants(boxsize_cMpc = 114.)

#Make the boxes
xcube, dcube, dtcube, z = c2t.freq_box(xfrac_dir, density_dir, z_low=7.0, cube_slices=500)

#Plot the dT box
pl.imshow(dtcube[0,:,:], extent=[z.min(),z.max(), 0, c2t.conv.LB], aspect='auto')
pl.xlabel('$z$')
pl.ylabel('Mpc')
pl.show()
