#This file illustrates some basic usage for the c2raytools package
#The script reads some data files and prints and plots some statistics

import c2raytools as c2t
import numpy as np
import pylab as pl

#Some path names. Modify these as needed
base_path = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
density_filename = base_path+'/coarser_densities/halos_removed/8.515n_all.dat'
xfrac_filename = base_path + '/114Mpc_f2_10S_256/results_ranger/xfrac3d_8.515.bin'
velocity_filename = base_path+'/coarser_densities/halos_removed/8.515v_all.dat'

#Enable the printing of various messages
c2t.utils.set_verbose(True)

#We are using the 114/h Mpc simulation box, so set all the proper conversion factors
c2t.conv.set_sim_constants(boxsize_cMpc = 114.)


#Read a density file and print some statistics
dfile = c2t.files.DensityFile(density_filename)

print 'The redshift is ', dfile.z
print 'The size of the mesh is (', dfile.mesh_x, dfile.mesh_y, dfile.mesh_z, ')'
print 'The mean baryon density is ', dfile.cgs_density.mean(), ' g/cm^3'

#Read an ionized fractions file
xfile = c2t.files.XfracFile(xfrac_filename)

print 'The volume-averaged mean ionized fraction is: ', xfile.xi.mean()
print 'The mass-averaged mean ionized fractions is: ', c2t.statistics.mass_weighted_mean_xi(xfile.xi, dfile.raw_density)

#Read a velocity data file
vfile = c2t.files.VelocityFile(velocity_filename)

#Since the velocity data is actually momentum, we need the density to convert it to km/s 
kms = vfile.get_kms_from_density(dfile)
print 'Gas velocity at cell (100,100,100) is ', kms[:,100,100,100], 'km/s'

#Calculate neutral hydrogen number density
n_hi = dfile.cgs_density*xfile.xi/c2t.const.m_p

#Calculate differential brightness temperature
dT = c2t.misc.calc_dt(xfile, dfile)

#Get a slice through the center
dT_slice = dT[128,:,:]

#Convolve with a Gaussian beam, assuming a 2 km maximum baseline
dT_slice_conv = c2t.instrumental.beam_convolve(dT_slice, xfile.z, c2t.conv.boxsize, max_baseline=2000.) 

#Plot some stuff
pl.figure()

pl.subplot(221)
pl.imshow(n_hi[128,:,:])
pl.colorbar()
pl.title('$n_{HI}$')

pl.subplot(222)
pl.imshow(dT_slice)
pl.colorbar()
pl.title('dT')

pl.subplot(223)
pl.imshow(dT_slice_conv)
pl.colorbar()
pl.title('dT (convolved)')

pl.show()
