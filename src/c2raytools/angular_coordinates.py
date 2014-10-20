'''
Created on Sep 17, 2014

@author: Hannes Jensen

Methods to convert data between physical (cMpc) coordinates
and observational (angular-frequency) coordinates
'''

import numpy as np
import lightcone
import cosmology as cm
import conv
import helper_functions as hf
import misc
import const

def physical_lightcone_to_observational(physical_lightcone, data_z_low, \
            data_box_width=None, output_nu_low=115., output_dtheta=1.17, \
            output_dnu=0.5, interp_order=2):
    '''
    Convert a lightcone in physical (length) units into observational (angle,
    frequency) units. The output volume will be padded to make the FoV the
    same all along the LOS axis. This means that periodic boundary conditions
    are not preserved. The LOS axis must be the last index.
    
    Parameters:
        * physical_lightcone (numpy array): the lightcone in physical units
        * data_z_low (float): the lowest redshift of the lightcone
        * data_box_width (float): the width of the lightcone in cMpc
            If set to None, the current value of conv.LB will be used
        * output_nu_low (float): the lowest frequency of the output in MHz 
        * output_dtheta (float): the angular resolution of the output in arcmin
        * output_dnu (float): the frequency resolution of the output in MHz
        * interp_order (int): the order to use for the spline interpolation (default 2)

        
    Returns:
        The observational lightcone with the specified angular and frequency 
        resolution.
    '''
    
    #Depth of the lightcone
    nb = float(physical_lightcone.shape[2])/float(physical_lightcone.shape[1])
    mx = physical_lightcone.shape[1] #Grid size along perp axis
    n_cells_los = nb*mx 
    if data_box_width == None:
        data_box_width = conv.LB

    #Calculate frequencies and redshifts along the LOS axis
    z_high = cm.nu_to_z(output_nu_low)
    lc_redshifts = lightcone.redshifts_at_equal_comoving_distance(data_z_low, z_high, \
                                            box_grid_n=mx, box_length_mpc=data_box_width)
    lc_frequencies = cm.z_to_nu(lc_redshifts)

    #FoV angular size at all lc_redshifts, in arcmin. 
    angle = cm.angular_size_comoving(data_box_width, lc_redshifts)*60.

    #Redshift indices for the highest and lowest frequencies
    idx_high_nu = 0 
    idx_low_nu = np.argmin(np.abs(lc_frequencies-output_nu_low)) 
    
    #Angle at highest frequency (worst angular resolution in lightcone) 
    fov_high = angle[idx_high_nu]
    dtheta_high = fov_high/mx
    
    #Width of the padded volume
    LB_padded = np.deg2rad(fov_high/60.)*cm.luminosity_distance(lc_redshifts[idx_low_nu])\
                /(1.+lc_redshifts[idx_low_nu])
    mx_padded = int(LB_padded/(data_box_width/mx))

    #Create and fill the new, larger light cone cuboid
    lightcone_padded = np.zeros((mx_padded, mx_padded, n_cells_los))
    lightcone_padded[0:mx,0:mx,:] = physical_lightcone
    lightcone_padded[mx:mx_padded,0:mx,:] = physical_lightcone[0:mx_padded-mx,:,:]
    lightcone_padded[0:mx,mx:mx_padded,:] = physical_lightcone[:,0:mx_padded-mx,:]
    lightcone_padded[mx:mx_padded,mx:mx_padded,:] = physical_lightcone[0:mx_padded-mx,0:mx_padded-mx,:]

    #Recalculate the FoV(z) for the new cuboid
    angle = cm.angular_size_comoving(LB_padded, lc_redshifts)*60.

    #Find the array size for the regularly spaced angular array theta
    nx = int(fov_high/dtheta_high)

    #Set the regularly spaced angular array theta
    theta = np.arange(nx)/(nx-1.0)*fov_high

    #First interpolate each redshift slice to angular coordinates
    new_box_length = idx_low_nu-idx_high_nu
    theta_lightcone = np.zeros((nx, nx, new_box_length))

    #Interpolate to the same regularly spaced angular array theta.
    #The input data has a different angular array for each redshift
    #Step through all required lc_redshifts
    for iz in np.arange(idx_high_nu, idx_low_nu):
        #angles along the perpendicular axis at this redshift
        thetaz = np.arange(mx_padded)/(mx_padded-1.)*angle[iz]
        #interpolation indices
        ith = hf.find_idx(thetaz, theta)
        #interpolate
        theta_lightcone[:,:,iz-idx_high_nu] = misc.interpolate2d(lightcone_padded[:,:,iz],\
                                                   ith, ith, order=interp_order)

    #Interpolate to regular frequency grid (at max resolution)
    #The input data is on an irregular frequency grid
    output_frequencies = np.arange(new_box_length)/float(new_box_length-1) *\
        (lc_frequencies[idx_low_nu]-lc_frequencies[idx_high_nu])+lc_frequencies[idx_high_nu]
    ifr = hf.find_idx(lc_frequencies, output_frequencies)
    output_lightcone = misc.interpolate3d(theta_lightcone, np.arange(nx), \
                                  np.arange(nx), ifr, order=interp_order)

    #Regrid to required resolution: dnu MHz, dtheta arcmin
    nfr = round((lc_frequencies[idx_high_nu]-lc_frequencies[idx_low_nu])/output_dnu)
    nth = round(angle[idx_low_nu]/output_dtheta)
    output_lightcone = misc.resample_array(output_lightcone, newdims=(nth,nth,nfr))
    
    return output_lightcone

    

def observational_lightcone_to_physical(observational_lightcone, input_nu_low, \
                                        input_dnu, input_dtheta, interp_order=2):
    '''
    Convert a lightcone in observational (angle, frequency) units
    to physical units (Mpc). The LOS axis must be the last index.
    
    Parameters:
        * observational_lightcone (numpy array): the input data with units
            in arcminutes and MHz
        * input_nu_low (float): the lowest frequency of the input data
        * input_dnu (float): the width of the input data cells in arcminutes
        * input_dtheta (float): the depth of the input data cells MHz
        * interp_order (int): the order to use for the spline interpolation (default 2)
        
    Returns:
        Tuple with (lightcone, redshifts, width) in physical coordinates 
        (constant comoving size).
        The resolution will be set by the lowest frequency.
    '''
    
    #Dimensions of input data (frequency, angle)
    n_input_freqs = observational_lightcone.shape[2] 
    n_input_theta = observational_lightcone.shape[1] 

    #Frequencies and redshifts of input data
    input_frequencies = np.arange(n_input_freqs)*input_dnu + input_nu_low
    input_redshifts = cm.nu_to_z(input_frequencies)

    input_fov = n_input_theta*input_dtheta 

    #----- angular direction

    #Find the luminosity distance (Mpc) at all FG redshifts
    input_lumdist = cm.luminosity_distance(input_redshifts)

    #Find for each redshift the comoving size (Mpc) of the FoV
    input_fov_mpc = np.deg2rad(input_fov/60.)*input_lumdist/(1.+input_redshifts)

    #Find the comoving size at low and high frequency ends
    fov_mpc_low = input_fov_mpc[0]
    fov_mpc_high = input_fov_mpc[n_input_freqs-1]

    #Set comoving size to the one at high frequency
    output_mpc_perp = fov_mpc_high

    #Set the resolution to the one at low frequency
    output_resolution_perp = fov_mpc_low/n_input_theta

    #---- frequency direction

    #Find the redshift dependent Hubble parameter
    Hz = const.Hz(input_redshifts)

    #Find the comoving size of every los cell
    input_cell_size_los = input_dnu/const.nu0*const.c/Hz*(1.0+input_redshifts)**2

    #Find the irregularly spaced coordinate of the los axis
    input_cell_position_los = np.zeros(n_input_freqs)
    for ifr in range(n_input_freqs): 
        input_cell_position_los[ifr] = np.sum(input_cell_size_los[0:ifr+1])
        
    #Find the comoving length of the los axis
    output_depth_mpc = np.sum(input_cell_size_los)

    #Set the los resolution to the largest cell in the los direction
    output_cell_size_los = np.max(input_cell_size_los)

    #---- construct the lowest resolution grid

    #Now find the worst resolution between the perpendicular and los directions
    output_cell_size = np.max([output_cell_size_los, output_resolution_perp])

    #Find the length the regularized los axis should have
    n_output_cells_los = np.round(output_depth_mpc/output_cell_size)

    #Find the array size
    n_output_cells_perp = np.round(output_mpc_perp/output_cell_size)

    #Create the regularized comoving los axis
    output_cell_position_los = np.linspace(0., output_depth_mpc, n_output_cells_los)
    start_distance = cm.z_to_cdist(input_redshifts[-1])
    output_z = cm.cdist_to_z(start_distance+output_cell_position_los)[::-1]

    #Set the comoving physical distance array (perpendicular direction)
    output_cell_position_perp = np.linspace(0., output_mpc_perp, n_output_cells_perp) 

    #First, interpolate perpendicular axis
    output_volume_perp = np.zeros((n_output_cells_perp,\
                                    n_output_cells_perp, n_input_freqs))
    for iz in range(n_input_freqs):
        xy_perp = np.linspace(0, input_fov_mpc[iz], n_input_theta) 
        ixy_perp = hf.find_idx(xy_perp, output_cell_position_perp)
        output_volume_perp[:,:,iz] = misc.interpolate2d(observational_lightcone[:,:,iz],\
                                                         ixy_perp, ixy_perp, order=interp_order)
        
    #Then, interpolate along frequency axis
    ilos = hf.find_idx(input_cell_position_los, output_cell_position_los)
    ixy_perp = np.arange(n_output_cells_perp)
    output_volume = misc.interpolate3d(output_volume_perp, \
                                       ixy_perp, ixy_perp, ilos, order=interp_order)
    
    return output_volume, output_z, output_mpc_perp




