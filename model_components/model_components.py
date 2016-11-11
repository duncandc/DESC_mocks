"""
Module containing classes used in DESC mocks
"""

from __future__ import (division, print_function, absolute_import)

import numpy as np
from scipy.interpolate import interp1d

from lss_observations.stellar_mass_functions import Tomczak_2014_phi
from .stellar_mass_functions import LiWhite_2009_phi, MBII_Desc_Phi, Illustris_Desc_Phi

__all__ = ['RankSmHm', 'SSFR', 'ConditionalGalaxyProps2D']
__author__=['Duncan Campbell']

class RankSmHm(object):
    """
    class to model the stellar mass-halo mass (SMHM) relation as a rank order relation
    between galaxies and haloes with scatter
    """
    def __init__(self,
                 prim_haloprop_key = 'halo_mpeak',
                 stellar_mass_function = 'LiWhite_2009',
                 Lbox = 250.0,
                 scatter = 0.2,
                 redshift = 0.0,
                 **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string
            key indicating the halo property used to calculate halo ranks
        
        stellar_mass_function : string
            string indicating which stellar mass function should be reproduced.
            one of the following: LiWhite_2009, MBII, Illustris
        
        Lbox : float
            length of one side of the simulation cube to populate (h^-3 Mpc^3)
        """
        
        self._mock_generation_calling_sequence = ['assign_stellar_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('stellar_mass', 'f4')])
        self.list_of_haloprops_needed = [prim_haloprop_key]
        self._prim_haloprop_key = prim_haloprop_key
        
        #calculate the volume of the simulation (h^-3 Mpc^3)
        self._volume = Lbox**3.0
        self.param_dict = {'scatter':scatter}
        self.redshift = redshift
        
        #initialize the stellar mass function
        if stellar_mass_function == 'LiWhite_2009':
            self.mstar_phi = LiWhite_2009_phi(**kwargs)
        elif stellar_mass_function == 'MBII':
            self.mstar_phi = MBII_Desc_Phi(**kwargs)
        elif stellar_mass_function == 'Illustris':
            self.mstar_phi = Illustris_Desc_Phi(**kwargs)
        elif stellar_mass_function == 'Tomczak_2014':
            self.mstar_phi = Tomczak_2014_phi(**kwargs)
        else:
            msg = ("stellar mass function not recognized.")
            raise ValueError(msg)
    
    def assign_stellar_mass(self,  **kwargs):
        """
        assign stellar masses to (sub-)haloes based on rank
        """
        
        table = kwargs['table']
        
        #sort haloes by prim_halo_prop
        mass = np.log10(table[self._prim_haloprop_key])
        halo_sort_inds = np.argsort(mass)
        
        #remove haloes that are not to be populated
        if 'remove' in table.keys():
            mask = (table['remove'][halo_sort_inds]==0)
        else:
            mask = np.array([True]*len(table))
        
        N_haloes_to_fill = np.sum(mask)
        
        #MC realization of stellar masses
        mstar = self.mc_sample_stellar_mass_func(N_haloes_to_fill, self._volume)
        galaxy_sort_inds = np.argsort(mstar)
        
        if self.param_dict['scatter']>0.0:
            #add scatter in steps.  choose step size
            sub_scatter = self.param_dict['scatter']/5.0
            
            #find where the log mass changes by the scatter
            mass = np.log10(mstar)
            i = (mass[galaxy_sort_inds]/sub_scatter).astype('int')
            dummy, inds = np.unique(i, return_index=True)
            
            #calculate the change in index corresponding to the change in mass
            dn = np.log10(np.diff(inds))
            n = np.log10((inds[1:]+inds[:-1])/2.0)
            dn[0] = 0.0
            
            fn = interp1d(n, dn, fill_value='extrapolate')
            
            sigma = 10.0**fn(np.log10(np.arange(0,len(table))))
            sigma = 1.0*sigma/len(table)
            
            for i in range(0, 5**2):
                galaxy_sort_inds = scatter_ranks(galaxy_sort_inds, sigma)
            
        #assign stellar masses based on rank
        table['stellar_mass'] = -99 #filler value
        table['stellar_mass'][halo_sort_inds[mask]] = mstar[galaxy_sort_inds]
    
    def mc_sample_stellar_mass_func(self, N, V):
        """
        Monte Carlo sample a stellar mass function in the specified volume, ``V``,
        the specified number fo times, ``N``.
        
        Parameters
        ----------
        N : int
            integer number of times to sample the stellar mass function
        
        V : float
            volume in h^-3 Mpc^3
        
        Returns
        -------
        mstar : numpy.array
            array of stellar masses
        """
        
        f = self.calculate_cumulative_mass_function(10.0**1.0, 10.0**13.5, self.mstar_phi, V)
        
        #calculate the minimum stellar mass
        min_mstar = f(np.log10(N))
        if min_mstar<0.0: min_mstar=0.0
        
        f = self.calculate_cumulative_mass_function(10.0**min_mstar, 10.0**13.5, self.mstar_phi, V)
        
        #sample mass function to get stellar masses
        s = np.log10(np.random.random(N)*N)
        mstar = 10**f(s)
        
        return mstar
    
    def calculate_cumulative_mass_function(self, min, max, phi, volume):
        """
        Calculate the cumulative stellar mass function, N(>mstar).
        
        Parameters
        ----------
        min : float
            minimum stellar mass
        
        max : float
            maximum stellar mass
        
        phi : function object
            number density per dex as a function of stellar mass
        
        volume : float
            volume in h^-3Mpc^3
        
        Return
        ------
        N(>m) : function object
           callable cumulative stellar mass function
        
        Notes
        -----
        The cumulative stellar mass function is calculated by numerically integrating the 
        differential stellar mass function.
        """
        
        #integrate to get cumulative distribution
        dm_log = 0.0001
        mstar_sample = np.arange(np.log10(min),np.log10(max),dm_log)
        mstar_sample = 10**mstar_sample[::-1]
        
        dN = phi(mstar_sample)*dm_log*(volume)
        cumulative_dist = np.cumsum(dN)
        
        #interpolate mas function
        f = interp1d(np.log10(cumulative_dist), np.log10(mstar_sample), fill_value='extrapolate')
        
        return f


class SSFR(object):
    """
    class to implement conditional abundance matching (CAM) method 
    to model the specific star-formation rate (SSFR) of galaxies.
    """
    def __init__(self, reference_galaxy_catalog,
                       stellar_mass_bins,
                       stellar_mass_key = 'stellar_mass',
                       prim_haloprop_key = 'halo_vpeak',
                       reference_ssfr_key = 'ssfr',
                       reference_stellar_mass_key = 'stellar_mass',
                       sigma_ssfr = 0.0,
                       inverse_correaltion = True,
                       **kwargs):
        """
        Parameters
        ----------
        reference_galaxy_catalog : astropy.table
            reference catalog used to sample the conditional ssfr-stellar 
            mass distribution
        
        stellar_mass_bins : array_like
            bins of stellar mass in which to assign ssfr
            and enforce correlation strength
        
        stellar_mass_key :  string, optional
            keyword for stellar mass property in mock
        
        prim_haloprop_key : string, optional
            keyword of halo property to use for CAM
        
        reference_ssfr_key : string, optional
            key into `reference_galaxy_catalog` which returns stellar mass SSFR
        
        reference_stellar_mass_key : string, optional
            key into `reference_galaxy_catalog` which returns stellar mass
        
        sigma_ssfr : float, optional
            correlation strength parameter between ``halo_prop`` and ssfr 
            at fixed stellar_mass.
        
        inverse_correaltion : boolean, optional
            enforce an inverse correlation 
        """
        
        self._mock_generation_calling_sequence = ['assign_ssfr']
        self._galprop_dtypes_to_allocate = np.dtype([('ssfr','f4')])
        self.list_of_haloprops_needed = [prim_haloprop_key]
        
        #extract necessary columns from reference catalog
        self._ref_ssfr = reference_galaxy_catalog[reference_ssfr_key]
        self._ref_stellar_mass = reference_galaxy_catalog[reference_stellar_mass_key]
        
        #set other parameters for model
        self._stellar_mass_bins = stellar_mass_bins
        self._prim_haloprop_key = prim_haloprop_key
        self._stellar_mass_key = stellar_mass_key
        self._new_galprop_name = 'ssfr'
        
        #define model parameters
        self.param_dict = ({'sigma_ssfr': sigma_ssfr,
                            'inverse_correlation': True})
    
    def assign_ssfr(self, **kwargs):
        """
        assign SSFRs to galaxies
        """
        
        table = kwargs['table']
        
        if 'remove' in table.keys():
            mask = (table['remove']==0)
        else:
            mask = np.array([True]*len(table))
        
        #insert filler value
        table[self._new_galprop_name] = -99
        
        #define stellar mass bins
        bins = self._stellar_mass_bins
        
        #bin galaxies by stellar mass
        mock_bin_inds = np.digitize(table[self._stellar_mass_key], bins=bins)
        mock_bin_inds[mask==False] = -1
        ref_bin_inds = np.digitize(self._ref_stellar_mass, bins=bins)
        
        mock_bin_inds = mock_bin_inds.astype('int')
        ref_bin_inds = ref_bin_inds.astype('int')
        
        #loop over stellar mass bins
        Nbins = len(bins)
        for i in range(1, Nbins):
            
            #get indicies of mock galaxies in the bin
            mock_inds_in_bin = np.where(mock_bin_inds==i)[0]
            mock_inds_in_bin = np.random.permutation(mock_inds_in_bin)
            ref_inds_in_bin = np.where(ref_bin_inds==i)[0]
            
            #if there are no galaxies, skip the bin
            N_mock_in_bin = len(mock_inds_in_bin)
            if N_mock_in_bin == 0:
                continue
            
            ssfrs = np.random.choice(self._ref_ssfr[ref_inds_in_bin], size=N_mock_in_bin)
            ssfrs = np.sort(ssfrs, kind='mergesort')
            
            if self.param_dict['sigma_ssfr'] == np.inf:
                sats = (table[self._halo_upid_key][mock_inds_in_bin]!=-1)
                table[self._new_galprop_name][mock_inds_in_bin] = ssfrs
            else:
                #sort the mock galaxies in the bin by vpeak
                sort_inds = np.argsort(table[self._prim_haloprop_key][mock_inds_in_bin], kind='mergesort')
                mock_inds_in_bin = mock_inds_in_bin[sort_inds]
                
                if self.param_dict['inverse_correlation']:
                    mock_inds_in_bin = mock_inds_in_bin[::-1]
                
                if self.param_dict['sigma_ssfr']==0.0:
                    #assign SSFRs
                    table[self._new_galprop_name][mock_inds_in_bin] = ssfrs
                else:
                    #add scatter to ssfr-vpeak correlation
                    mock_inds_in_bin = scatter_ranks(mock_inds_in_bin, self.param_dict['sigma_ssfr'])
                    #assign SSFRs
                    table[self._new_galprop_name][mock_inds_in_bin] = ssfrs


class ConditionalGalaxyProps2D(object):
    """
    class to carry over conditional galaxy properties to a mock
    from a reference galaxy catalogue
    """
    def __init__(self,
                 reference_galaxy_catalogue,
                 prim_galprops = {'stellar_mass':np.logspace(9.5,12.0,25),'ssfr':np.linspace(8.0,12.5,0.1)},
                 galprops_to_allocate = ['absmag_g', 'absmag_r'],
                 ref_key_translation = None,
                 **kwargs):
        """
        Parameters
        ----------
        reference_galaxy_catalogue : astropy.table object
            reference galaxy catalogue
        
        prim_galprops : dictionary
            length 2 dictionary of primary galaxy properties and bins
            
        galprops_to_allocate : list
            new galaxy properties to assign
            
        ref_key_translation : dictionary
            dictionary providing the translation between keys in the mock to keys in the 
            reference galaxy catalogue.  If None, the keys are assumed to be the same
        """
        
        #build translation dictionary
        if ref_key_translation is None:
            ref_key_translation = [key for key in prim_galprops.keys()]
            for key in galprops_to_allocate:
                ref_key_translation.append(key)
            ref_key_translation = dict([key,key] for key in ref_key_translation)
        for value in ref_key_translation.values():
            if not value in reference_galaxy_catalogue.dtype.names:
                msg = ('key not in reference galaxy catalogue.')
                raise ValueError(msg)
        self._ref_key_translation = ref_key_translation
        
        #get dtypes of new galaxy properties
        dtypes = []
        for key in galprops_to_allocate:
            t = reference_galaxy_catalogue[ref_key_translation[key]].dtype.descr[0][1]
            dtypes.append((key,t))
        
        self._mock_generation_calling_sequence = ['assign_new_properties']
        self._galprop_dtypes_to_allocate = np.dtype(dtypes)
        self._galprops_to_allocate = galprops_to_allocate
        self.list_of_haloprops_needed = []
        
        self.prim_galprops = prim_galprops
        self.prim_galprop_keys = prim_galprops.keys()
        
        self.reference_catalogue = reference_galaxy_catalogue
        
        #bin the galaxies in the reference catalogue
        xbins = self.prim_galprops[self.prim_galprop_keys[0]]
        ybins = self.prim_galprops[self.prim_galprop_keys[1]]
        ref_x = self.reference_catalogue[ref_key_translation[self.prim_galprop_keys[0]]]
        ref_y = self.reference_catalogue[ref_key_translation[self.prim_galprop_keys[1]]]
        self.ref_bin_ids = self.get_bin_id(ref_x, ref_y, xbins, ybins)
    
    def assign_new_properties(self, **kwargs):
        """
        draw from the conditional galaxy property distribution to assign
        new galaxy properties to mock galaxies
        """
        
        table = kwargs['table']
        
        #bin the mock galaxies
        xbins = self.prim_galprops[self.prim_galprop_keys[0]]
        ybins = self.prim_galprops[self.prim_galprop_keys[1]]
        mock_x = table[self.prim_galprop_keys[0]]
        mock_y = table[self.prim_galprop_keys[1]]
        bin_ids = self.get_bin_id(mock_x, mock_y, xbins, ybins)
        
        inds = match_draw(bin_ids, self.ref_bin_ids)
        mask = (inds!=-1)
        for key in self._galprops_to_allocate:
            table[key] = -99
            table[key][mask] = self.reference_catalogue[self._ref_key_translation[key]][inds[mask]]
        
    def get_bin_id(self, x, y, xbins, ybins):
        """
        return an ID number indicatig which bin each galaxy falls witin
        
        Parameters
         ----------
        x, y: numpy.array
             array of values
         
        xbins, ybins : numpy.array
            bins
            
        Returns
        -------
        bin_ids : numpy.array
            length x array of interger IDs corresponding to the 2-D bin
        """
        
        x_inds = np.digitize(x, bins=xbins)
        y_inds = np.digitize(y, bins=ybins)
        
        Nx = len(xbins)-1
        Ny = len(ybins)-1
        N = Nx*Ny
        idxy = np.arange(0,N).reshape((Nx,Ny))
        
        mask = ((x_inds>0) & (x_inds<Nx)) & ((y_inds>0) & (y_inds<Ny))
        
        ids = np.zeros((len(x),), dtype='int')-1
        ids[mask] = idxy[x_inds[mask],y_inds[mask]]
        
        return ids


def match_draw(ar1, ar2, replace=True):
    """
    Return matches in ``ar2`` of ``ar1`` where ``ar1`` and ``ar2`` are both in general 
    non-unique arrays of integers, and the matches in ``ar2`` are drawn randomly.
    
    Parameters
    ----------
    ar1 : array_like
        array of integers for which to find matches in ``ar2``
    
    ar2 : array_like
        array of integers to search for matches
    
    replace : bool, optional
        boolean indictaing whether matches should be drawn with replacement in ``ar2``.
        
    Returns
    -------
    idx : numpy.array
        *len(ar1)* length array of indices into ``ar2`` that match ``ar1``.  
        If no matches in ``ar2`` are found for an element in ``ar1``, a *-1* is returned
        at that position
    """
    
    #sort input arrays
    sort_inds_1 = np.argsort(ar1)
    sort_inds_2 = np.argsort(ar2)
    ar1 = ar1[sort_inds_1]
    ar2 = ar2[sort_inds_2]
    
    #create arrays to undo argsort
    orig_inds_1 = np.arange(0,len(ar1)).astype('int')
    orig_inds_2 = np.arange(0,len(ar2)).astype('int')
    unsort_inds_1 = orig_inds_1[sort_inds_1]
    unsort_inds_2 = orig_inds_2[sort_inds_2]
    
    #create array to store result
    result = np.zeros(len(ar1), dtype='int')-1
    
    #find which values are in each array
    unq_ar1, Nar1 = np.unique(ar1, return_counts=True)
    unq_ar2, Nar2 = np.unique(ar2, return_counts=True)
    
    uniq_all = np.unique(np.hstack((unq_ar1,unq_ar2)))
    
    mask1 = np.in1d(uniq_all, unq_ar1)
    mask2 = np.in1d(uniq_all, unq_ar2)
    
    N1 = np.zeros(len(uniq_all)).astype('int')
    N2 = np.zeros(len(uniq_all)).astype('int')
    
    N1[mask1] =  Nar1
    N2[mask2] =  Nar2
    
    if replace==True:
        i1 = 0
        i2 = 0
        for i, id in enumerate(uniq_all):
            if N1[i]>0:
                if N2[i]>0:
                    result[sort_inds_1[i1:i1+N1[i]]] = np.random.choice(sort_inds_2[i2:i2+N2[i]],N1[i], replace=True)
            i1 = i1 + N1[i]
            i2 = i2 + N2[i]
    else:
        i1 = 0
        i2 = 0
        for i, id in enumerate(uniq_all):
            if N1[i]>0:
                if N2[i]>=N1[i]:
                    result[sort_inds_1[i1:i1+N1[i]]] = sort_inds_2[i2:i2+N1[i]]
                elif N2[i]>1:
                    result[sort_inds_1[i1:i1+N2[i]]] = sort_inds_2[i2:i2+N2[i]]
            i1 = i1 + N1[i]
            i2 = i2 + N2[i]
    
    return result


def scatter_ranks(arr, sigma):
    """
    Scatter the index of values in an array.
    
    Parameters
    ----------
    arr : array_like
        array of values to scatter
        
    sigma : array_like
        scatter relative to len(arr) 
    
    Returns
    -------
    scatter_array : numpy.array
        array with same values as ``arr``, but the locations of those values 
        have been scatter.
    """
    
    sigma = np.atleast_1d(sigma)
    if len(sigma)==1:
        sigma = np.repeat(sigma, len(arr))
    elif len(sigma)!=len(arr):
        raise ValueError("sigma must be same length as ``arr``.")
    
    #get array of indicies before scattering
    N = len(arr)
    inds = np.arange(0,N)
    
    mask = (sigma>1000.0)
    sigma[mask] = 1000.0
    
    #get array to scatter positions
    mask = (sigma>0.0)
    dn = np.zeros(N)
    dn[mask] = np.random.normal(0.0,sigma[mask]*N)
    
    #get array of new indicies
    new_inds = inds + dn
    new_inds = np.argsort(new_inds, kind='mergesort')
    
    return arr[new_inds]