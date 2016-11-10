#!/usr/bin/env python

#Duncan Campbell
#August 2016
#Yale University
#examine SMHM relation in mocks

#load packages
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py

from model_components import LiWhite_2009_phi, MBII_Desc_Phi, Illustris_Desc_Phi

from halotools import sim_manager
from halotools.empirical_models import SubhaloModelFactory

from astropy.table import Table

#get project directory
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    
    #open mock
    mock_name = 'yale_cam_age_matching_LiWhite_2009_z0.0'
    filename = dir_path + 'data/mocks/' + mock_name + '.hdf5'
    f = h5py.File(filename, 'r')
    mock = f.get(mock_name)
    mock = Table(np.array(mock))
    
    #examine SMHM relation
    cens = (mock['halo_upid']==-1)
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    p = plt.scatter(mock['halo_mvir'][cens], mock['stellar_mass'][cens], s=2,
                    c=mock['ssfr'][cens], lw=0, alpha=1.0,
                    vmin = -12.5, vmax = -9.0,
                    cmap = 'jet_r', rasterized=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$m_{\rm vir}$')
    plt.ylabel(r'$m_{*}$')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$\log({\rm ssfr})$')
    plt.show()
    
    x = np.log10(mock['halo_mvir'][cens])
    y = np.log10(mock['stellar_mass'][cens])
    bins = np.arange(10,15,0.2)
    
    #examine scatter in SMHM relaiton
    from halotools.mock_observables import mean_y_vs_x
    bin_centers, rho, sigma = mean_y_vs_x(x, y,bins=bins, error_estimator='variance')
    
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.plot(bin_centers, sigma)
    plt.plot(bin_centers, bin_centers*0.0 + 0.17, '--')
    plt.xlabel(r'$m_{\rm vir}$')
    plt.ylabel(r'$\sigma$')
    plt.show()

if __name__ == '__main__':
    main()