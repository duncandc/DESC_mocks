#!/usr/bin/env python

#Duncan Campbell
#August 2016
#Yale University
#examine color magnitude diagram in mocks

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
    mock_name = 'yale_cam_age_matching_LiWhite_2009'
    filename = dir_path + 'data/mocks/' + mock_name + '.hdf5'
    f = h5py.File(filename, 'r')
    mock = f.get(mock_name)
    mock = Table(np.array(mock))
    
    #examine color-magnitude diagram
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    p = plt.scatter(mock['absmag_r'], mock['g-r'], s=2,
                    c=mock['ssfr'], lw=0, alpha=1.0,
                    vmin = -12.5, vmax = -9.0,
                    cmap = 'jet_r', rasterized=True)
    plt.xlim([-16,-23])
    plt.ylim([0,1.5])
    plt.xlabel(r'$M_r - 5\log(h)$')
    plt.ylabel(r'$g-r$')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$\log({\rm ssfr})$')
    plt.show()

if __name__ == '__main__':
    main()
