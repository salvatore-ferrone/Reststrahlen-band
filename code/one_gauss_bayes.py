import pymc
import numpy as np 
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

    wave_number_range = wavenumber.max() - wavenumber.min()
    upper_threshold = wave_number_range/4
    
    lower_threshold = 5*np.mean(np.diff(wavenumber))
    with pymc.Model() as model:
        m = pymc.Normal('m', mu=prior_m_mu, sigma=prior_m_sigma)
        b = pymc.Normal('b', mu=prior_b_mu, sigma=prior_b_sigma)
        A=pymc.Normal('A', mu=A_mu, sigma=A_sigma)
        kbar=pymc.Normal('kbar', mu=Kbar, sigma=Kbar_sigma)
        sigma=pymc.HalfNormal('sigma', sigma=prior_sigma_mu)
        
        pymc.Potential('upper_bound', pymc.math.switch(sigma > upper_threshold, -np.inf, 0))
        pymc.Potential('lower_bound', pymc.math.switch(sigma < lower_threshold, -np.inf, 0))
        my_model_spectrum=model_single_gauss_spec(wavenumber,b,m,A,kbar,sigma)
        observed = spectrum
        Y_obs = pymc.Normal('Y_obs', mu=my_model_spectrum, sigma=float(noise_amplitude), observed=spectrum)
        
    return model

        
def variable_names():
    return ["m","b","A","kbar","sigma"]
        
        
        
def load_paths(paths,survey_name,model_name,row_index):
    #### PATHS
    path_to_spectra=paths.otes_csv(survey_name)
    path_to_wave_numbers=paths.wavenumbers
    out_file_name=paths.bayes_fits_fname(survey_name,row_index,model_name)
    return path_to_spectra,path_to_wave_numbers,out_file_name

def load_target_spectrum(path_to_spectra,path_to_wave_numbers,wv_num_min,wv_num_max,row_index):
    spectra_csv, wave_nb=FOS.open_data(path_to_spectra,path_to_wave_numbers)
    wavenumber, spectra_filter=FOS.filter_by_wavenumber(wave_nb,wv_num_min,wv_num_max)
    sclk=spectra_csv.iloc[row_index]['sclk_string']
    my_spectrum=spectra_csv.iloc[row_index][spectra_filter].to_numpy()
    my_spectrum=np.array(my_spectrum,dtype=float)
    return wavenumber,my_spectrum,sclk


def main(row_index,survey_name):
    
    
    start_time = datetime.datetime.now()
    #### PARAMETERS
    model_name  =   "one_gauss"
    paths       =   PathHandler()
    wv_num_min, wv_num_max, chain_length,chain_sample,n_chains=FOS.set_parameters()
    variable_names=FOS.get_variable_names()
    
    #### PATHS
    path_to_spectra,path_to_wave_numbers,out_file_name=load_paths(paths,survey_name,model_name,row_index)
    if os.path.exists(out_file_name):
        print("Skipping\n",out_file_name, "\n It already exists\nDelete if you wish recompute")
        return
    
    #### LOAD TARGET SPECTRUM 
    wavenumber,my_spectrum,sclk=load_target_spectrum(path_to_spectra,path_to_wave_numbers,wv_num_min,wv_num_max,row_index)

    
    ### PART 1 DONE
    print("Data loaded",datetime.datetime.now()-start_time)
    
    ##### DO THE MARKOV CHAIN
    model=build_model(wavenumber,my_spectrum)
    print("Model built",datetime.datetime.now()-start_time)
    ### DO  TRACE
    with model:
        trace=pymc.sample(chain_length)
    ### Get number of divergences
    num_diverge=np.sum(trace.sample_stats.diverging.to_numpy())
    print("Trace done",datetime.datetime.now()-start_time)
    
    ### STORE IN DICTIONARY 
    trace_dict=PBF.store_trace_in_dict(sclk,variable_names,trace,chain_length,chain_sample)
    outarray=np.zeros((len(variable_names),chain_sample*n_chains))
    for i in range(len(variable_names)):
        outarray[i,:]=trace_dict[sclk][variable_names[i]]
    print("Trace stored",datetime.datetime.now()-start_time)
    #### WRITE TO HDF5
    FOS.write_to_hdf5(\
        out_file_name,
        survey_name,
        row_index,
        outarray,
        variable_names,
        sclk,
        n_chains,
        chain_sample,
        chain_length,
        wavenumber,
        my_spectrum,
        wv_num_min,
        wv_num_max,
        path_to_spectra,
        path_to_wave_numbers,
        num_diverge,
        model_name)
    print("Data saved",datetime.datetime.now()-start_time)
        
    
    
    
if __name__ == "__main__":
    row_index=int(sys.argv[1])
    survey_name=sys.argv[2]
    main(int(sys.argv[1]),sys.argv[2])