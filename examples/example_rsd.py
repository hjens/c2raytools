'''
This example shows how to make a dT-box with peculiar velocity distortions
(redshift space distortions) applied. It also calculates the spherically-averaged
power spectrum.
'''

import c2raytools as c2t
import numpy as np
import pylab as pl

#Some path names. Modify as needed to point to directories
#that are available on your system.
base_path = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
density_filename = base_path+'/coarser_densities/halos_removed/30.000n_all.dat'
velocity_filename = base_path+'/coarser_densities/halos_removed/30.000v_all.dat'

#Enable verbose output
c2t.set_verbose(True)

#We are using the 114/h Mpc simulation box, so set all the proper conversion factors
c2t.set_sim_constants(boxsize_cMpc = 114.)

#Read density
dfile = c2t.DensityFile(density_filename)

#Read velocity data file and get the actual velocity 
vfile = c2t.VelocityFile(velocity_filename)
kms = vfile.get_kms_from_density(dfile)

#To speed things up, we will assume that the IGM is completely neutral, so instead
#of reading an ionization fraction file, we will just make an array of zeros
xi = np.zeros_like(dfile.cgs_density)

#Calculate the dT
dT_realspace = c2t.calc_dt(xi, dfile, z=dfile.z)

#Make the redshift-space volume
dT_redshiftspace = c2t.get_distorted_dt(dT_realspace, kms, \
                                 dfile.z, los_axis=0, num_particles=20)

#Calculate spherically-averaged power spectra, 
#using 20 logarithmically spaced k bins, from 1e-1 to 10
kbins = 10**np.linspace(-1, 1, 20)
ps_dist, k = c2t.power_spectrum_1d(dT_redshiftspace, kbins)
ps_nodist, k = c2t.power_spectrum_1d(dT_realspace, kbins)

#Plot ratio. On large scales, this should be close to 1.83
pl.semilogx(k, ps_dist/ps_nodist)
pl.xlabel('$k \; \mathrm{[Mpc^{-1}]}$')
pl.ylabel('$P_k^{\mathrm{PV}}/P_k^{\mathrm{NoPV}}$')
pl.show()
