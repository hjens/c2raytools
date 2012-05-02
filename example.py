import c2raytools as c2t
import numpy as np
import pylab as pl

#Some path names. Modify these as needed
base_path = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
density_filename = base_path+'/coarser_densities/halos_removed/8.515n_all.dat'
xfrac_filename = base_path + '/114Mpc_f2_10S_256/results_ranger/xfrac3d_8.515.bin'

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
