import pymc
import numpy as np 
import pandas as pd
import h5py 
import perform_bayes_fit as PBF
import sys 
import datetime
from path_handler import PathHandler
import os 
import perform_bayes_fit as PBF 
import fit_one_spectrum as FOS 



def model_single_gauss_spec(x,b,m,A,kbar,sigma):
    continuum = PBF.continuum(x,b,m)
    band = PBF.gaussian(x,A,kbar,sigma)
    return continuum-band


def build_model(wavenumber, spectrum):
    
    
    (prior_b_mu,prior_b_sigma), \
    (prior_m_mu,prior_m_sigma),\
    (A_mu,A_sigma),\
    (Kbar,Kbar_sigma),\
    (prior_sigma_mu,prior_sigma_sigma),\
    noise_amplitude = \
        PBF.obtain_single_band_spectrum_priors(wavenumber, spectrum)

    
    with pymc.Model() as model:
        m = pymc.Normal('m', mu=prior_m_mu, sigma=prior_m_sigma)
        b = pymc.Normal('b', mu=prior_b_mu, sigma=prior_b_sigma)
        A=pymc.Normal('A', mu=A_mu, sigma=A_sigma)
        kbar=pymc.Normal('kbar', mu=Kbar, sigma=Kbar_sigma)
        sigma=pymc.HalfNormal('sigma', sigma=prior_sigma_sigma)
        
        my_model_spectrum=model_single_gauss_spec(wavenumber,b,m,A,kbar,sigma)
        observed = spectrum
        Y_obs = pymc.Normal('Y_obs', mu=my_model_spectrum, sigma=float(noise_amplitude), observed=spectrum)
        
    return model
        
        
def main(row_index,survey_name):
    
    paths=PathHandler()
    start_time = datetime.datetime.now()
    # open params
    wv_num_min, wv_num_max, chain_length,chain_sample,n_chains=\
        FOS.set_parameters()
    # import 
    path_to_spectra=paths.otes_csv(survey_name)
    path_to_wave_numbers=paths.wavenumbers
    out_file_name=paths.bayes_fits_fname(survey_name,row_index)
    
    # check if outfile alrady exists
    if os.path.exists(out_file_name):
        print("Skipping\n",out_file_name, "\n It already exists\nDelete if you wish recompute")
        return
    
    spectra_csv, wave_nb=\
        FOS.open_data(path_to_spectra,path_to_wave_numbers)
    wavenumber, spectra_filter=\
        FOS.filter_by_wavenumber(wave_nb,wv_num_min,wv_num_max)
    variable_names=FOS.get_variable_names()
    
    sclk=spectra_csv.iloc[row_index]['sclk_string']
    my_spectrum=spectra_csv.iloc[row_index][spectra_filter].to_numpy()
    my_spectrum=np.array(my_spectrum,dtype=float)
    
    
    print("Data loaded",datetime.datetime.now()-start_time)