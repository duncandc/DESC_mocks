#!/usr/bin/env python

#Duncan Campbell
#August 2016
#Yale University
#examine stellar mass function of mocks

#load packages
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py

#from ..model_components.model_components import LiWhite_2009_phi, MBII_Desc_Phi, Illustris_Desc_Phi
from lss_observations.stellar_mass_functions import Tomczak_2014_phi, LiWhite_2009_phi

from halotools import sim_manager
from halotools.empirical_models import SubhaloModelFactory

from astropy.table import Table

#get project directory
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))+'/../'

def main():
    
    #open mock
    mock_name = 'yale_cam_age_matching_LiWhite_2009_z0.00'
    #mock_name = 'yale_cam_age_matching_MBII_z0.00'
    #mock_name = 'yale_cam_age_matching_Illustris_z0.00'
    #mock_name = 'yale_cam_age_matching_Tomczak_2014_z0.00'
    filename = dir_path + 'data/mocks/' + mock_name + '.hdf5'
    f = h5py.File(filename, 'r')
    mock = f.get(mock_name)
    mock = Table(np.array(mock))
    
    print(mock.dtype.names)
    
    def stellar_mass_func(mock):
        """
        caclulate stellar mass function
        """
        
        #stellar mass function
        bins = np.arange(6.0,13.0,0.1)
        bins = 10.0**bins
        bin_centers = (bins[:-1]+bins[1:])/2.0
        counts = np.histogram(mock['stellar_mass'],bins=bins)[0]
        dndm = counts/(100.0**3)/0.1
        
        return dndm, bin_centers
    
    dndm_1, bin_centers = stellar_mass_func(mock)
    
    if mock_name.split('_')[-2] == '2009':
        comparison_phi = LiWhite_2009_phi()
    elif mock_name.split('_')[-2] == 'MBII':
        comparison_phi = MBII_Desc_Phi()
    elif mock_name.split('_')[-2] == 'Illustris':
        comparison_phi = Illustris_Desc_Phi()
    elif mock_name.split('_')[-2] == '2014':
        comparison_phi = Tomczak_2014_phi()
        comparison_phi = LiWhite_2009_phi()
    
    comparison_m = np.logspace(7,12,100)
    comparison_dndm = comparison_phi(comparison_m)
    
    #plot stellar mass function
    fig = plt.figure(figsize=(6.6,6.6))
    
    #upper panel
    rect = 0.2,0.35,0.7,0.55
    ax = fig.add_axes(rect)
    p0, = ax.plot(bin_centers, comparison_phi(bin_centers), '-', color='red')
    p1, = ax.plot(bin_centers, dndm_1, '-', color='black')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\phi(M_{*})~[h^{5}{\rm Mpc}^{-3}M_{\odot}^{-1}]$', labelpad=-1)
    ax.set_xlim([10**7,10**13])
    ax.set_ylim([10**-14,10**0])
    ax.xaxis.set_visible(False)

    #lower panel
    rect = 0.2,0.2,0.7,0.15
    ax = fig.add_axes(rect)
    ax.plot([0,10**15],[0,0],'-',color='red')
    ax.plot(bin_centers, (dndm_1-comparison_phi(bin_centers))/comparison_phi(bin_centers),'-', color='black')
    ax.set_ylim([-0.25,0.25])
    ax.set_yticks([-0.2,0.0,0.2])
    ax.set_ylabel(r'$\Delta\phi/\phi_{\rm SDSS}$', labelpad=-2)
    ax.set_xlabel(r'$M_{*} ~[h^{-2}M_{\odot}]$')
    ax.set_xscale('log')
    ax.set_xlim([10**7,10**13])
    
    plt.show()


if __name__ == '__main__':
    main()