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
    
    filename = 'nyu_lss_mpa_vagc_dr7_bbright'
    savepath = dir_path + 'data/'
    savename = "sdss_reference_catalogue"
    
    #open sdss galaxy catalogue
    f =  h5py.File(dir_path + 'data/' + filename + '.hdf5', 'r')
    GC = f.get(filename) #halo catalogue
    GC = np.array(GC)
    
    #dictionary translating keys from those stored in the galaxy catalogue
    #to those I want to use in the construction of the mock
    key_translation = {'stellar_mass':'sm_MEDIAN',
                       'ssfr':'sfr_MEDIAN',
                       'absmag_g':'ABSMAG_g.none.model.z0.00',
                       'absmag_r':'ABSMAG_r.none.model.z0.00'}
    
    #define cosmology
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    
    #trim the catalogue a bit
    zmin = 0.01
    zmax = 0.2
    selection = (GC['M']<=17.6) & (GC['Z']<zmax) & (GC['Z']>zmin) &\
                (GC['ZTYPE']==1) & (GC['FGOTMAIN']>0)
    GC = GC[selection]
    
    #apply h scaling
    GC[key_translation['stellar_mass']] = GC[key_translation['stellar_mass']]+np.log10(0.7**2.0)
    
    #sample redshift range of objects
    z = np.linspace(zmin,zmax,1000)
    #calculate luminosity distance as a function of redshift
    dL = cosmo.luminosity_distance(z).value
    #build interpolation function
    fdL = interpolate.interp1d(z,dL,kind='linear')
    
    #calculate stellar mass completeness limit, van den Bosch et al. 2008
    Mstar_lim = (4.852 + 2.246*np.log10(dL) + 1.123*np.log10(1+z) - 1.186*z)/(1.0-0.067*z)
    
    #make completeness cut to catalogue
    Mstar = GC[key_translation['stellar_mass']]
    dL = fdL(GC['Z']) #luminosity distance of catalogue objects
    z = GC['Z'] #redshift
    LHS = (4.852 + 2.246*np.log10(dL) + 1.123*np.log10(1+z) - 1.186*z)/(1.0-0.067*z)
    
    keep = (Mstar>LHS)
    GC = GC[keep] #only keep objects above completeness limit
    
    print("total number of galaxies in sample: ",len(GC))
    
    #create new array with only the desired quantities
    data = Table()
    for key in key_translation.keys():
        data[key] = GC[key_translation[key]]
    
    data['stellar_mass'] = 10**data['stellar_mass']
    #data['absmag_g'] =data['absmag_g'] - 5.0 * np.log10(0.7)
    #data['absmag_r'] =data['absmag_r'] - 5.0 * np.log10(0.7)
    
    print(data)
    
    #save result
    print("saving result as: ", savepath+savename+'.hdf5')
    data.write(savepath+savename+'.hdf5', format='hdf5', path = savename, overwrite=True)


if __name__ == '__main__':
    main()