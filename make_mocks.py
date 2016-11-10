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

from model_components import RankSmHm, SSFR, ConditionalGalaxyProps2D

from halotools import sim_manager
from halotools.empirical_models import SubhaloModelFactory

from astropy.table import Table

#get project directory
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    
    np.random.seed(seed=0)
    
    #define mock parameters
    redshift = 0.0 #0.0, 0.0750029562581298, 0.745566261695294
    prim_haloprop_key = 'halo_vpeak'
    secondary_haloprop_key = 'halo_half_mass_scale'
    stellar_mass_function = 'LiWhite_2009' #'LiWhite_2009', 'MBII', 'Illustris', 'Tomczak_2014'
    log_min_mstar = 8.5
    sigma_smhm = 0.15
    
    savepath = dir_path + 'data/mocks/'
    savename = 'yale_cam_age_matching_'+stellar_mass_function + '_z'+"{:.2f}".format(redshift)
    print(savename)
    
    #load halo catalogue
    filename = 'data/MassiveBlack/hlist_'+"{:.5f}".format(1.0/(1.0+redshift))+'.list.hdf5'
    fname = dir_path + filename
    halocat = convert_to_halocat(fname)
    print('\n halo catalogue columns:')
    for name in halocat.halo_table.dtype.names: print('     '+name)
    
    #open reference sdss galaxy catalogue
    filename = 'data/sdss_reference_catalogue.hdf5'
    fname = dir_path + filename
    f =  h5py.File(fname, 'r')
    reference_catalog = f.get('sdss_reference_catalogue') #halo catalogue
    reference_catalog = Table(np.array(reference_catalog))
    print('\n reference galaxy catalogue columns:')
    for name in reference_catalog.dtype.names: print('     '+name)
    
    print(np.min(reference_catalog['stellar_mass']))
    
    #define stellar mass model component
    mstar_model = RankSmHm(prim_haloprop_key = prim_haloprop_key,
                           stellar_mass_function = stellar_mass_function,
                           Lbox=halocat.Lbox, redshift=0.0, scatter=sigma_smhm)
    
    #define specific star-formation model component
    haloprop_key = secondary_haloprop_key
    mstar_bins = 10.0**np.arange(log_min_mstar,12.0,0.1)
    ssfr_model = SSFR(reference_catalog, mstar_bins,
                      reference_ssfr_key = 'ssfr',
                      reference_stellar_mass_key = 'stellar_mass',
                      prim_haloprop_key = haloprop_key)
    ssfr_model.param_dict['sigma_ssfr'] = 0.0
    ssfr_model.param_dict['inverse_correlation'] = False
    
    #define model for propagating additional galaxy properties
    prim_galprops = {'stellar_mass':mstar_bins,'ssfr':np.linspace(-8.0,-12.5,46)}
    galprops_to_allocate = ['absmag_g', 'absmag_r']
    conditional_props = ConditionalGalaxyProps2D(reference_catalog, prim_galprops, galprops_to_allocate)
    
    #define galaxy selection
    def galaxy_selection_func(table):
        mask = (table['stellar_mass'] >= 10**log_min_mstar)
        return mask
    
    composite_model = SubhaloModelFactory(stellar_mass = mstar_model,
                                          ssfr = ssfr_model,
                                          conditional_props = conditional_props,
                                          model_feature_calling_sequence = ('stellar_mass', 'ssfr', 'conditional_props'),
                                          galaxy_selection_func = galaxy_selection_func,
                                          )
    
    #populate simulation
    composite_model.populate_mock(halocat = halocat)
    mock = composite_model.mock.galaxy_table
    
    #calculate galaxy color
    mock['g-r'] = (mock['absmag_g'] - mock['absmag_r'])
    print(mock.dtype.names)
    
    mock.write(savepath+savename+'.hdf5', format='hdf5', path=savename, overwrite=True)


def convert_to_halocat(fname):
    """
    Convert a processed halotools halo catalog h5py file w/ meta data to
    a halotools halocat object.
    
    This purpose of this function is to avoid having to add the halo catalogue to cache.
    
    Parameters
    ----------
    filename of halo catalogue
    """
    
    f = h5py.File(fname, 'r')
    
    halo_table_dict = {key:f['data'][key] for key in f['data'].dtype.names}
    metadata_dict =  {key:f.attrs[key] for key in f.attrs.keys()}
    
    d = dict(halo_table_dict.items() + metadata_dict.items())
    d['redshift'] = float(d['redshift'])
    
    halocat = sim_manager.UserSuppliedHaloCatalog(**d)
    
    return halocat

if __name__ == '__main__':
    main()