#!/usr/bin/env python

#Duncan Campbell
#March, 2015
#Yale University
#Process Rockstar halo catalogue

#load packages
from __future__ import print_function, division
import numpy as np

import h5py
import sys
import re

from halotools import sim_manager

def main():
    
    #filenames and filepaths
    input_fpath = './data/MassiveBlack/'
    output_fpath = './data/MassiveBlack/'
    
    if len(sys.argv)==1:
        input_fname = 'hlist_1.00000.list'
    else:
        input_fname =  sys.argv[1]
    output_fname = input_fname+'.hdf5'
    
    #set some properties of the simulation
    scale_factor = float(re.findall("\d+\.\d+",input_fname)[0])
    redshift = 1.0/scale_factor - 1.0
    Lbox = 100.0 # h^-1 Mpc
    particle_mass = 1.1*10**7 # h^-1 Msol
    simname = 'MassiveBlack_100'
    halo_finder='Rockstar'
    version_name = '1.0'
    
    columns_to_keep_dict = {'halo_id': (1, 'i8'),
                            'halo_pid': (5, 'i8'),
                            'halo_upid': (6, 'i8'),
                            'halo_mvir': (10, 'f4'),
                            'halo_rvir': (11, 'f4'),
                            'halo_x': (17, 'f4'),
                            'halo_y': (18, 'f4'),
                            'halo_z': (19, 'f4'),
                            'halo_vx': (20, 'f4'),
                            'halo_vy': (21, 'f4'),
                            'halo_vz': (22, 'f4'),
                            'halo_m200b': (39, 'f4'),
                            'halo_mpeak': (61, 'f4'),
                            'halo_vpeak': (63, 'f4'),
                            'halo_half_mass_scale': (64, 'f4'),
                            'halo_mpeak_scale': (70, 'f4'),
                            'halo_acc_scale': (71, 'f4'),
                            'halo_first_acc_scale': (72, 'f4'),
                            'halo_vmax_at_mpeak': (75, 'f4'),
                            }
    
    #apply cuts to catalogue
    row_cut_min_dict = {'halo_mpeak': particle_mass*100}
    processing_notes = 'catalog only contains (sub-)halos with mpeak mass greater than 100 particles.'
    
    #read in catalogue and save results
    reader = sim_manager.RockstarHlistReader(input_fpath+input_fname, columns_to_keep_dict,\
        output_fpath+output_fname, simname, halo_finder, redshift, version_name, Lbox, particle_mass,\
        row_cut_min_dict=row_cut_min_dict, processing_notes=processing_notes,\
        overwrite=True) 
    
    columns_to_convert_from_kpc_to_mpc = ['halo_rvir']
    reader.read_halocat(columns_to_convert_from_kpc_to_mpc,\
        write_to_disk = True, update_cache_log = True) 


if __name__ == '__main__':
    main()