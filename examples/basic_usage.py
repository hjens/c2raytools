'''
This file illustrates some of the most basic usage for the c2raytools package.
It reads some data files and prints and plots some statistics. 
To run this script, you must have c2raytools installed in a location where
Python can find it. To do this, see the Readme file, or http://ttt.astro.su.se/~hjens/c2raytools/
Second, you must modify the path names to point to files that you actually have access to.

For more information, see the full documentation at 
http://ttt.astro.su.se/~hjens/c2raytools/
'''

import c2raytools as c2t
import numpy as np
import pylab as pl

#Some path and file names. Modify these as needed.
base_path = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
density_filename = base_path+'/coarser_densities/nc256_halos_removed/8.515n_all.dat'
xfrac_filename = base_path + '/114Mpc_f2_10S_256/results_ranger/xfrac3d_8.515.bin'
velocity_filename = base_path+'/coarser_densities/nc256_halos_removed/8.515v_all.dat'

#Enable the printing of various messages
c2t.set_verbose(True)

#We are using the 114/h Mpc simulation box, so set all the proper conversion factors.
#Be sure to always set this before loading anything. Otherwise, c2raytools will
#not know how to convert densities and velocities to physical units!
c2t.set_sim_constants(boxsize_cMpc = 114.)


#Read a density file and store it as a DensityFile object.
dfile = c2t.DensityFile(density_filename)

#The density file object contains various useful properties, such as the 
#redshift, the mesh size and the actual density. 
print 'The redshift is ', dfile.z
print 'The size of the mesh is (', dfile.mesh_x, dfile.mesh_y, dfile.mesh_z, ')'
print 'The mean baryon density is ', dfile.cgs_density.mean(), ' g/cm^3'
print 'The critical baryon density of the universe is: ', c2t.rho_crit_0*c2t.OmegaB, ' g/cm^3'

#You can also access the density in simulation units
print 'The raw density at point (0,0,0) is ', dfile.raw_density[0,0,0]

#Read an ionized fractions file and store it as an XfracFile object
xfile = c2t.XfracFile(xfrac_filename)

#The most important property of an XfracFile object is xi, which
#is a numpy array containing the ionized fraction
print 'The ionized fraction in point (10,223,45) is: ', xfile.xi[10,223,45]
print 'The volume-averaged mean ionized fraction is: ', xfile.xi.mean()

#c2raytools has several methods to calculate useful statistics, such as 
#mass-weighted mean ionized fraction
print 'The mass-averaged mean ionized fraction is:', c2t.mass_weighted_mean_xi(xfile.xi, dfile.raw_density)

#Read a velocity data file and store it as a VelocityFile object
vfile = c2t.VelocityFile(velocity_filename)

#Since the velocity data is actually momentum, we need the density to convert it to km/s 
kms = vfile.get_kms_from_density(dfile)
print 'Gas velocity at cell (100,100,100) is ', kms[:,100,100,100], 'km/s'

#Calculate neutral hydrogen number density
n_hi = dfile.cgs_density*xfile.xi/c2t.m_p

#Calculate differential brightness temperature
#The calc_dt method can also take names of files so you don't have to load the
#files yourself in advance.
dT = c2t.calc_dt(xfile, dfile, z=xfile.z)

#Get a slice through the center
dT_slice = dT[128,:,:]

#Convolve with a Gaussian beam, assuming a 2 km maximum baseline
dT_slice_conv = c2t.beam_convolve(dT_slice, xfile.z, fov_mpc=c2t.conv.LB, \
                                  max_baseline=2000.)


#c2raytools comes with a few simple plotting functions to quickly 
#visualize data. For example, to plot a slice through the ionization
#fraction data, you can simply do:
c2t.plot_slice(xfile) 

#If you want more control over the plotting, you have to use, e.g.,
#matplotlib directly
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
