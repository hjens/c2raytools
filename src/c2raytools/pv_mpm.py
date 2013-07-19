import numpy as np
import const
import conv
from helper_functions import print_msg, get_interpolated_array


def get_distorted_dt(dT, kms, redsh, los_axis=0, num_particles=10):
	''' 
	Apply peculiar velocity distortions to a differential
	temperature box, using the Mesh-Particle-Mesh method,
	as described in http://arxiv.org/abs/1303.5627
	
	Parameters:
		* dT (numpy array): the differential temperature box
		* kms (numpy array): velocity in km/s, array of dimensions 
			(3,mx,my,mz) where (mx,my,mz) is dimensions of dT
		* redsh (float): the redshift
		* los_axis = 0 (int): the line-of-sight axis (must be 0, 1 or 2)
		* num_particles = 10 (int): the number of particles to use per cell
			A higher number gives better accuracy, but worse performance.
		
	Returns:
		The redshift space box as a numpy array with same dimensions as dT.
		
	Example:
		Read a density file, a velocity file and an xfrac file, calculate the 
		brightness temperature, and convert it to redshift space.
		
		>>> vfile = c2t.VelocityFile('/path/to/data/8.515v_all.dat')
		>>> dfile = c2t.DensityFile('/path/to/data/8.515n_all.dat')
		>>> xfile = c2t.XfracFile('/path/to/data/xfrac3d_8.515.bin')
		>>> dT = c2t.calc_dt(xfile, dfile)
		>>> kms = vfile.get_kms_from_density(dfile)
		>>> dT_zspace = get_distorted_dt(dT, dfile, dfile.z, los_axis = 0)
	'''


	#Take care of different LOS axes
	assert (los_axis == 0 or los_axis == 1 or los_axis == 2)
	if los_axis == 0:
		get_slice = lambda data, i, j : data[:,i,j]
	elif los_axis == 1:
		get_slice = lambda data, i, j : data[i,:,j]
	else:
		get_slice = lambda data, i, j : data[i,j,:]

	#Dimensions
	mx,my,mz = dT.shape

	print_msg('Making velocity-distorted box...')
	print_msg('The redshift is %.3f' % redsh)
	print_msg('The box size is %.3f cMpc' % conv.LB)
	
	#Figure out the apparent position shift 
	vpar = kms[los_axis,:,:,:]
	z_obs= (1+redsh)/(1.-vpar/const.c)-1.
	dr = (1.+z_obs)*kms[los_axis,:,:,:]/const.Hz(z_obs)

	#Make the distorted box
	distbox = np.zeros((mx,my,mz))
	part_dT = np.zeros(mx*num_particles)

	last_percent = 0
	for i in range(my):
		percent_done = int(float(i)/float(my)*100)
		if percent_done%10 == 0 and percent_done != last_percent:
			print_msg('%d %%' % percent_done)
			last_percent = percent_done
		for j in range(mz):

			#Take a 1D slice from the dT box
			dT_slice = get_slice(dT,i,j)

			#Divide slice into particles
			partpos = np.linspace(0,conv.LB,mx*num_particles) #Positions before dist.
			for n in range(num_particles): #Assign dT to particles
				part_dT[n::num_particles] = dT_slice/float(num_particles)

			#Calculate and apply redshift distortions
			cell_length = conv.LB/float(mx)
			dr_slice_pad= get_slice(dr,i,j)
			np.insert(dr_slice_pad,0,dr_slice_pad[-1])
			dr_slice = get_interpolated_array(dr_slice_pad, len(partpos), 'linear')
			dr_slice = np.roll(dr_slice,num_particles/2)
			partpos += dr_slice

			#Boundary conditions
			partpos[np.where(partpos < 0)] += conv.LB
			partpos[np.where(partpos > conv.LB)] -= conv.LB

			#Regrid particles
			dist_slice = np.histogram(partpos, bins=np.linspace(0,conv.LB,mx+1), weights = part_dT)[0]
			if los_axis == 0:
				distbox[:,i,j] = dist_slice
			elif los_axis == 1:
				distbox[i,:,j] = dist_slice
			else:
				distbox[i,j,:] = dist_slice

	print_msg('Old dT (mean,var): %3f, %.3f' % ( dT.mean(), dT.var()) )
	print_msg('New (mean,var): %.3f, %.3f' % (distbox.mean(), distbox.var()) )
	return distbox


