import numpy as np
from .. import const
from .. import utils
from .. import conv

def get_distorted_dt(dT, kms, redsh, los_axis=0, num_particles=10):
	''' Apply peculiar velocity distortions to a differential
	temperature box, using the Mesh-Particle-Mesh method
	dT - the temperature box
	kms - velocity in km/s, array of dimensions (3,mx,my,mz) where (mx,my,mz) is dimensions of dT
	redsh - the redshift
	los_axis - the line-of-sight (must be 0, 1 or 2) (default 0)
	num_particles - the number of particles to use per cell (default 10)
	return distorted box with same dimensions as dT '''


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

	utils.print_msg('Making velocity-distorted box...')
	utils.print_msg('The redshift is %.3f' % redsh)
	utils.print_msg('The box size is %.3f cMpc' % conv.LB)
	
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
			utils.print_msg('%d %%' % percent_done)
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
			dr_slice = utils.get_interpolated_array(dr_slice_pad, len(partpos), 'linear')
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

	utils.print_msg('Old dT (mean,var): %3f, %.3f' % ( dT.mean(), dT.var()) )
	utils.print_msg('New (mean,var): %.3f, %.3f' % (distbox.mean(), distbox.var()) )
	return distbox


