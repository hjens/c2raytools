#This example shows how to make dT-box with peculiar velocity distortions applied

import c2raytools as c2t
import numpy as np
import pylab as pl

#Some path names. Modify these as needed
base_path = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
density_filename = base_path+'/coarser_densities/halos_removed/30.000n_all.dat'
velocity_filename = base_path+'/coarser_densities/halos_removed/30.000v_all.dat'

#Enable output
c2t.set_verbose(True)

#We are using the 114/h Mpc simulation box, so set all the proper conversion factors
c2t.set_sim_constants(boxsize_cMpc = 114.)

#Read density
dfile = c2t.DensityFile(density_filename)

#Read a velocity data file
vfile = c2t.VelocityFile(velocity_filename)
kms = vfile.get_kms_from_density(dfile)

#Make a distorted box. Assume x_i = 0. We could of course also have calculated dT like in example.py
#and passed that to get_distorted_dt
distorted = c2t.get_distorted_dt(dfile.raw_density.astype('float64'), kms, dfile.z, los_axis=0, num_particles=20)

#Calculate power spectra
ps_dist,k = c2t.power_spectrum_1d(distorted)
ps_nodist,k = c2t.power_spectrum_1d(dfile.raw_density)

#Plot ratio
pl.semilogx(k, ps_dist/ps_nodist)
pl.xlabel('$k \; \mathrm{[Mpc^{-1}]}$')
pl.ylabel('$P_k^{\mathrm{PV}}/P_k^{\mathrm{NoPV}}$')
pl.show()
