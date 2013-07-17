import numpy as np
import xfrac_file
import density_file
import conv
from helper_functions import get_data_and_type

def plot_slice(data, los_axis = 0, slice_num = 0, logscale = False, **kwargs):
    '''
    Plot a slice through a data cube.
    
    Parameters:
        * data (XfracFile, DensityFile, string or numpy array): the data to 
            plot. The function will try to determine what type of data it's 
            been given. 
        * los_axis = 0 (integer): the line of sight axis. Must be 0,1 or 2
        * slice_num = 0 (integer): the point along los_axis where the slice
            will be taken.
        * logscale = False (bool): whether to plot the logarithm of the data
            
    Kwargs:
        All kwargs are sent to matplotlib's imshow function. This can be used to,
        for instance, change the colormap.
        
    Returns:
        Nothing.
    '''
    
    import pylab as pl
    
    #Determine data type
    plot_data, datatype = get_data_and_type(data)
    
    #Take care of different LOS axes
    assert (los_axis == 0 or los_axis == 1 or los_axis == 2)
    if los_axis == 0:
        get_slice = lambda data, i : data[i,:,:]
    elif los_axis == 1:
        get_slice = lambda data, i : data[:,i,:]
    else:
        get_slice = lambda data, i : data[:,:,i]
    
    data_slice = get_slice(plot_data, slice_num)
    ext = [0, conv.LB, 0, conv.LB]
    if (logscale):
        data_slice = np.log10(data_slice)

    #Plot
    pl.imshow(data_slice, extent=ext, **kwargs)
    cbar = pl.colorbar()
    pl.xlabel('$\mathrm{cMpc}$')
    pl.ylabel('$\mathrm{cMpc}$')
    
    #Set labels etc
    if datatype == 'xfrac':
        if (logscale):
            cbar.set_label('$\log_{10} x_i$')
        else:
            cbar.set_label('$x_i$')
        pl.title('Ionized fraction')
    elif datatype == 'density':
        pl.title('Density')
        if (logscale):
            cbar.set_label('$\log_{10} \\rho \; \mathrm{[g \; cm^{-3}]}$')
        else:
            cbar.set_label('$\\rho \; \mathrm{[g \; cm^{-3}]}$')
        
        
if __name__ == '__main__':
    import c2raytools as c2t
    import pylab as pl
    
    dfilename = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/coarser_densities/nc256_halos_removed/6.905n_all.dat'
    #plot_slice('/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/114Mpc_f2_10S_256/results_ranger/xfrac3d_8.958.bin',)
    
    dfile = c2t.DensityFile(dfilename)
    plot_slice(dfile, los_axis=1, cmap=pl.cm.hot)
    
    
    
    