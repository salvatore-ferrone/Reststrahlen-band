import numpy as np
import scipy
import pymc as pymc
import os 
import pickle

############################################
############## DEFINE MODEL ################
############################################
def gaussian(x, A, mu, sigma):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
def gaussians(x, A, mu, sigma):
    assert len(A)==len(mu)
    assert len(A)==len(sigma)
    bands=np.zeros_like(x)
    for i in range(len(A)):
        bands+=gaussian(x, A[i], mu[i], sigma[i])
    return bands
def continuum(x, b, m,):
    return m*x+b

def model_spectrum(x,  b, m, A, mu, sigma):
    assert len(A)==len(mu)
    assert len(A)==len(sigma)
    bands=gaussians(x, A, mu, sigma)
    return continuum(x,b,m)-bands

############################################
###### FOR FINDING THE BAND-DEPTH ##########
############################################

def band_minimum_position_objective(x, A, mu, sigma):
    return -gaussians(x, A, mu, sigma)


############################################
############## PRIORS ######################
############################################
def obtain_single_band_spectrum_priors(wavenumber,spectrum):
    # put in 1-emissivity
    big_band=-(spectrum-spectrum.max())
    single_gauss_initial_guess=[0.5,wavenumber.mean(),np.std(wavenumber)]
    continuum_initial_guess=[1,0] # intercept, slope
    popt_band, pcov_band = scipy.optimize.curve_fit(gaussian, wavenumber, big_band, p0=single_gauss_initial_guess)
    popt_cont, _ = scipy.optimize.curve_fit(continuum, wavenumber, spectrum, continuum_initial_guess)
    ####################################
    ####### the big band priors ########
    ####################################
    ##### They are based on the max likelihood of the band
    
    # the band amplitude
    A_mu=popt_band[0]
    A_sigma=2*np.sqrt(pcov_band[0,0])
    # the band center
    Kbar=popt_band[1]
    Kbar_sigma=2*np.sqrt(pcov_band[1,1])
    # the band width
    prior_sigma_mu= popt_band[2]
    prior_sigma_sigma= 2*np.sqrt(pcov_band[2,2])
    
    ####################################
    ####### the continuum priors #######
    ####################################
    ###### It should be a flat line with 1. This is rather uninformed
    prior_b_mu = 1
    prior_m_mu = 0
    prior_m_sigma = 1 # this is a very wide prior, a slope of 1 is very steep
    
    # calculate where y-intercept would be by fitting a straight line through the spectrum
    #   then use this as the uncertainity in the y-intercept
    m=popt_cont[1]
    y=spectrum.mean()
    b=y-m*wavenumber.mean()
    prior_b_sigma = 5 # rather wide prior since we only have a gentle slope on the order of 1 + 1/1000
    
    ## the noise amplitude
    
    likelihood = model_spectrum(wavenumber, prior_b_mu, prior_m_mu, [A_mu], [Kbar], [prior_sigma_mu])
    
    noise=likelihood-spectrum
    
    noise_amplitude=np.std(noise)


    return (prior_b_mu,prior_b_sigma), (prior_m_mu,prior_m_sigma),(A_mu,A_sigma),(Kbar,Kbar_sigma),(prior_sigma_mu,prior_sigma_sigma),noise_amplitude
    
def obtain_multiple_band_spectrum_priors(wavenumber, spectrum, n_bands=2):
    """
    Obtain the prior parameters for multiple band spectra.

    Parameters:
    - wavenumber (array-like): The wavenumber values.
    - spectrum (array-like): The spectrum values.
    - n_bands (int, optional): The number of bands. Default is 2.

    Returns:
    - tuple: A tuple containing the prior parameters for each band:
        - (prior_b_mu, prior_b_sigma): The prior mean and standard deviation for b.
        - (prior_m_mu, prior_m_sigma): The prior mean and standard deviation for m.
        - (prior_A_mu, prior_A_sigma): The prior mean and standard deviation for A.
        - (prior_kbar, prior_kbar_sigma): The prior mean and standard deviation for kbar.
        - (prior_sigma_mu, prior_sigma_sigma): The prior mean and standard deviation for sigma.
    """
    assert n_bands == 2, "Only 2 bands are supported at the moment"
    
    # Obtain the prior parameters for a single band spectrum
    (prior_b_mu,prior_b_sigma), \
    (prior_m_mu,prior_m_sigma),\
    (A_mu,A_sigma),\
    (Kbar,Kbar_sigma),\
    (prior_sigma_mu,prior_sigma_sigma),\
    noise_amplitude = \
    obtain_single_band_spectrum_priors(wavenumber, spectrum)
    
    # Calculate the prior parameters for multiple bands
    kbar_left = Kbar - 1 * prior_sigma_mu
    kbar_right = Kbar + 1 * prior_sigma_mu
    kbar_sigma_left = 2 * Kbar_sigma
    
    prior_A_mu = np.ones(n_bands) * A_mu / n_bands
    prior_A_sigma = np.ones(n_bands) * A_sigma
    
    prior_kbar = np.array([kbar_left, kbar_right])
    prior_kbar_sigma = np.array([kbar_sigma_left, kbar_sigma_left])
    
    prior_sigma_mu = np.ones(n_bands) * prior_sigma_mu
    prior_sigma_sigma = np.ones(n_bands) * prior_sigma_sigma
    
    return (prior_b_mu, prior_b_sigma), \
    (prior_m_mu, prior_m_sigma), \
    (prior_A_mu, prior_A_sigma), \
    (prior_kbar, prior_kbar_sigma), \
    (prior_sigma_mu, prior_sigma_sigma), \
    noise_amplitude


############################################
############## LIKELIHOOD ##################
############################################
def build_pymc_model(wavenumber, spectrum, n_bands=2):
    """
    Build a model for a two-gaussian spectrum.

    Parameters:
    - wavenumber (array-like): The wavenumber values.
    - spectrum (array-like): The spectrum values.

    Returns:
    - function: A function that represents the model.
    """
  
    (prior_b_mu, prior_b_sigma), \
    (prior_m_mu, prior_m_sigma), \
    (prior_A_mu, prior_A_sigma), \
    (prior_kbar, prior_kbar_sigma), \
    (prior_sigma_mu, prior_sigma_sigma),\
    noise_amplitude=\
    obtain_multiple_band_spectrum_priors(wavenumber, spectrum)

    with pymc.Model() as model:
        A,kbar,sigma=[],[],[]
        m=pymc.Normal('m', mu=prior_m_mu, sigma=prior_m_sigma)
        b=pymc.Normal('b', mu=prior_b_mu, sigma=prior_b_sigma)
        for i in range(n_bands):
            A.append(pymc.Normal(f'A_{i}', mu=prior_A_mu[i], sigma=prior_A_sigma[i]))
            kbar.append(pymc.Normal(f'kbar_{i}', mu=prior_kbar[i], sigma=prior_kbar_sigma[i]))
            sigma.append(pymc.HalfNormal(f'sigma_{i}', sigma=prior_sigma_mu[i]))
        my_model_spectrum=model_spectrum(wavenumber, b, m, A, kbar, sigma)
        observed=spectrum
        Y_obs = pymc.Normal('Y_obs', mu=my_model_spectrum, sigma=float(noise_amplitude), observed=spectrum)
                        
    return model



###########################################
############ POST PROCESSING ##############
###########################################

def store_trace_in_dict(sclk, variables, trace, chain_length, chain_sample, nchains=4):
    """
    Store the trace of variables for a given sclk.
    
    We take the end of the number of markov chains samples as the posterior samples.

    Parameters:
    sclk (str): The sclk value.
    variables (list): A list of variables to store the trace for.
    trace (Trace): The trace object containing the posterior samples.
    chain_length (int): The length of the MCMC chain.
    chain_sample (int): The number of samples to take from the end of the chain.

    Returns:
    dict: A dictionary containing the stored traces for each variable.

    """
    traces = {}
    traces[sclk] = {}
    assert chain_length > chain_sample , "chain_sample must be less than chain_length"
    
    outlength = chain_sample*nchains
    for var in variables:
        outtemp = np.zeros(outlength)
        for i in range(nchains):
            downdex=i*chain_sample
            upindex=(i+1)*chain_sample
            outtemp[downdex:upindex] = trace.posterior[var].data[i, chain_length-chain_sample:chain_length]
            traces[sclk][var] = outtemp
    return traces


  
def convert_trace_to_dict(trace,chain_length):
    variables=list(trace.posterior.data_vars) 
    n_chain_sample=chain_length//2
    output={}
    most_probable_values = {}
    for var in variables:
        output[var]=trace.posterior.data_vars[var].to_numpy()[0,-n_chain_sample:]
        most_probable_values[var] = np.mean(output[var])
    return output, most_probable_values