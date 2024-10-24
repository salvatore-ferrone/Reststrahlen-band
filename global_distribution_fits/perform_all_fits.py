import numpy as np 
import matplotlib.pyplot as plt
import sys 
import h5py
import os
from scipy import stats
from scipy import optimize
sys.path.append("../code/")
from path_handler import PathHandler
import multiprocessing as mp
ph = PathHandler()

"""
This module is for fitting the global distributions of the band depth ratios
with a normal and gamma distribution. The idea is that we have a distribution
that is the result of a stochastic process + something else that is skewing it
we argue that the part skewing it is the result of hydration from the reststrahlen
band.
"""


def my_model_gamma(x, mu, sigma, shape, scale, loc, alpha):
    """
    The model
    """
    normal_part = (1 - alpha) * stats.norm.pdf(x, mu, sigma)
    gamma_part = alpha * stats.gamma.pdf(x, shape, scale=scale, loc=loc)
    return normal_part, gamma_part

def log_likelihood_gamma(params, data):
    """
        The likelihood
    """
    mu, sigma, shape, scale, loc, alpha = params
    if mu < 0:
        print(mu)
    if sigma < 0:
        print(sigma )
    if shape < 0 :
        print(shape)
    if loc < 0:
        print(loc)
    if alpha < 0:
        print(alpha)
    
    normal_part, gamma_part = my_model_gamma(data, mu, sigma, shape, scale, loc, alpha)
    return -np.sum(np.log(normal_part + gamma_part))


def constrain_guess_gauss_gamma(data):
    """
        Automatiting the guess of the initial conditions
    """
    numeric_threshold = 1e-10
    
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    shape_init,scale_init,loc_init=stats.gamma.fit(data)
    alpha_init = 0.5

    if loc_init < 0:
        loc_init = np.abs(loc_init)

    if scale_init < 0:
        scale_init = np.abs(scale_init)
    
    if shape_init < 0:
        shape_init = np.abs(shape_init)
    
    mu_bounds,sigma_bounds                  = (numeric_threshold, np.inf), (0, np.inf)
    shape_bounds,scale_bounds,loc_bounds    = (numeric_threshold, np.inf), (numeric_threshold, np.inf), (numeric_threshold, np.inf)
    alpha_bounds                            = (0, 1)
    # put them together
    bounds = [mu_bounds, sigma_bounds, shape_bounds, scale_bounds, loc_bounds, alpha_bounds]
    initial_guess = [mu_init, sigma_init, shape_init, scale_init, loc_init, alpha_init]
    return initial_guess,bounds
    


def obtain_quantified(i,ratio,basepath):
    """
        Fitting the data and writing out the results into a temp file
    """
    try:
        init_guess, bounds = constrain_guess_gauss_gamma(ratio)
        results = optimize.minimize(log_likelihood_gamma, init_guess, args=(ratio,), bounds=bounds, method='Nelder-Mead')
        mu, sigma, shape, scale, loc, alpha = results.x
        ANHYDROUS = alpha * stats.gamma.cdf(1, shape, scale=scale, loc=loc)
        HYDRATED = alpha * stats.gamma.sf(1, shape, scale=scale, loc=loc)
        tempdir = basepath + "myfit"+str(i)+".txt"
        myout=np.array([mu, sigma, shape, scale, loc, alpha, HYDRATED, ANHYDROUS])
        np.savetxt(tempdir, myout)
        return alpha, HYDRATED, ANHYDROUS
    except Exception as e:
        # Handle the exception (e.g., log the error and return a default value or re-raise the exception)
        print(f"Iteration {i}. An error occurred: {e} {i}")
        # Optionally, return default values or re-raise the exception
        return None, None, None

def model_switcher(survey_name):
    """
        The spectral fits either used one or two gaussian bands, depending on the survey
    """
    models={
        "EQ1":"two_gauss",
        "EQ2":"one_gauss",
        "EQ3":"two_gauss",
        "EQ6":"one_gauss"}
    return models[survey_name]


def open_band_depths(survey_name):
    """
        Open the data for EQ2 and EQ6
    """
    model_name=model_switcher(survey_name)
    inname = ph.band_depths_on_shape_model_fname(survey_name, model_name)
    with h5py.File(inname, 'r') as file:
        ratios = file['band_depths'][:,:]
    return ratios


def main(NCPU,numerator,denominator):

    assert numerator in ["EQ6","EQ2","EQ3","EQ1"]
    assert denominator in ["EQ6","EQ2","EQ3","EQ1"]

    # output name 
    foldername ="{:s}_divided_by_{:s}".format(numerator,denominator)
    basepath="/poubelle/sferrone/OREX/temp/"+foldername+"/"
    os.makedirs(basepath, exist_ok=True)
    filename = "band_depth_ratio_survival_gauss_gamma_fits.hdf5"
    outfname=ph.global_distributions(foldername,filename)

    print("Fitting {:s} divided by {:s}".format(numerator,denominator))
    print("will save to ", outfname)
    # Open the data and compute the ratio
    tasks = []
    numerator_data = open_band_depths(numerator)
    denominator_data = open_band_depths(denominator)
    n_trials = numerator_data.shape[1]
    for ii in range(n_trials):   
        ratio = numerator_data[:,ii]/denominator_data[:,ii]
        ratio = ratio[np.isfinite(ratio)]
        tasks.append((ii,ratio,basepath))

    print("Fitting {:d} trials".format(n_trials))
    # Run the fitting
    with mp.Pool(NCPU) as pool:
        results = pool.starmap(obtain_quantified, tasks)

    print("multiprocessing done")
    ii=0
    fname = basepath + "myfit{:d}.txt".format(ii)
    fit_params = np.loadtxt(fname)
    nparams = len(fit_params)
    surface_content_params,model_params  = np.zeros((n_trials, 3)), np.zeros((n_trials, nparams-2))
    


    for ii in range(n_trials):
        fname = basepath + "myfit{:d}.txt".format(ii)
        fit_parameters = np.loadtxt(fname)
        model_params[ii,:] = fit_parameters[0:(nparams-2)]
        surface_content_params[ii,:] = fit_parameters[(nparams-3):nparams]
    # change from survival to typical
    surface_content_params[:,0] = 1 - surface_content_params[:,0] 
    model_parameter_names = ["mu", "sigma", "shape", "scale", "loc", "alpha"]
    content_names = ["Typical", "Anhydrous", "Hydrated"]
    attrs = {"n_trials":n_trials,
            "author":"Salvatore Ferrone",
            "date":"2024-oct-23",
            "email":"salvatore.ferrone@uniroma1.it",
            "Model": "Gaussian + Gamma",}

    # put them in the file and have a good day
    myfile=h5py.File(outfname, 'w')
    myfile.create_dataset("surface_content_parameters", data=surface_content_params)
    myfile.create_dataset("surface_content_parameter_names", data=np.array(content_names, dtype='S'))
    myfile.create_dataset("model_parameter", data=model_params)
    myfile.create_dataset("model_parameter_names", data=np.array(model_parameter_names, dtype='S'))
    myfile.attrs.update(attrs)
    myfile.close()    

    print("saved output to", outfname)
    print("Done")


if __name__=="__main__":
    NCPU = 20
    numerator = "EQ3"
    denominator = "EQ2"
    main(NCPU,numerator,denominator)