#!/usr/bin/env python

#Duncan Campbell
#March, 2016
#Yale University
#make SDSS galaxy sample used to construct the mock

#load packages
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
from astropy.table import Table
from astropy.io.ascii import write

#get project directory
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    
    from astropy.io import ascii
    data1 = ascii.read(dir_path + 'data/hlsp_candels_hst_wfc3_uds_santini_v1_mass_cat.txt')
    data2 = ascii.read(dir_path + 'data/hlsp_candels_hst_wfc3_uds_santini_v1_physpar_cat.txt')
    
    z = np.array(data1['zbest'])
    mstar = np.array(data1['M_med'])
    sfr = np.array(data2['SFR_14a_deltau'])
    ssfr = np.array(sfr/mstar)
    u = np.array(data2['UMag_6a_tau'])
    v = np.array(data2['VMag_6a_tau'])
    j = np.array(data2['JMag_6a_tau'])
    
    print(ssfr)
    print(mstar)
    
    plt.figure()
    plt.scatter(mstar, ssfr, s=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([10**5,10**12])
    plt.ylim([10**(-12),10**(-6)])
    plt.show()
    
    plt.figure()
    plt.scatter(z, mstar, s=2,
                c=np.log10(np.array(ssfr)), lw=0, alpha=1.0,
                vmin = -12.5, vmax = -6.0,
                cmap = 'jet_r', rasterized=True)
    plt.yscale('log')
    plt.xlim([0,1])
    plt.ylim([10**5,10**12])
    plt.show()

if __name__ == '__main__':
    main()