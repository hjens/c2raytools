'''
This file demonstrates how to make a redshift-space
lightcone volume from many coeval volumes.
To do this, we first make separate light cones for
ionized fraction, density, brightness tempearture and velocity.
Note that this script is fairly memory-hungry, so run it on
a machine with plenty of RAM.
'''

import c2raytools as c2t
import numpy as np
import glob

c2t.set_verbose(True)
c2t.set_sim_constants(425)

#Paths to data files. Modify these as needed
xfrac_dir = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP5/425Mpc_WMAP5/f2_10S_504/results/'
density_dir = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP5/425Mpc_WMAP5/coarser_densities/'
velocity_dir = density_dir

#The place to save the results. Modify to something you have write access to
output_dir = '/home/hjens/links/local/slask/'

#Redshift limits for the light cone
z_low = 7.5
z_high = 12. 

#List all the redhifts for which we have data files
density_redshifts = c2t.get_dens_redshifts(density_dir, z_low, z_high, bracket=True)
xfrac_redshifts = c2t.get_xfrac_redshifts(xfrac_dir, z_low, z_high, bracket=True)

#List all the data files
dens_files = [density_dir + '%.3fn_all.dat' % z for z in density_redshifts]
vel_files = [velocity_dir + '%.3fv_all.dat' % z for z in density_redshifts]
xfrac_files = [xfrac_dir + 'xfrac3d_%.3f.bin' % z for z in xfrac_redshifts]

#Make the ionization fraction lightcone
xfrac_lightcone, z = c2t.make_lightcone(xfrac_files, z_low, z_high)

#Make the density lightcone
dens_lightcone, z = c2t.make_lightcone(dens_files, z_low, z_high)

#Combine ionization fraction and density to make a dT lightcone
dT_lightcone = c2t.calc_dt_lightcone(xfrac_lightcone, dens_lightcone, \
                                     lowest_z=z.min())

#Make a velocity lightcone
vel_lightcone, z = c2t.make_velocity_lightcone(vel_files, dens_files, \
                                               z_low, z_high)

#Apply redshift space distortions. This is done in the same way as for
#coeval data volumes. Just be sure to set periodic to False and 
#to specify the the velocity_axis argument (see the documentation
#of get_distorted_dt for more information) 
rsd_dT = c2t.get_distorted_dt(dT_lightcone, vel_lightcone, z, \
                              num_particles=30, los_axis=2, velocity_axis=0, \
                              periodic=False)

#Save the results
c2t.save_cbin(output_dir + 'lightcone_rsd.cbin', rsd_dT)