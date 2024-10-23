"""
Here, I will compute the deviation from gaussiantiy for the
    global distribtuion of the band depth of
    the Reststrahlen band.

MODEL: 


f(R|mu,sigma,a,shape,loc,alpha) = (1-alpha)*N(x|mu,sigma) + alpha*G(x|a,shape,loc)

    R: band depth of the RS band between the two different surveys
    N is the normal distribution 
    G is the gamma distribution
    alpha is the weighting factor between the two distributions    

*****************************************
From Paulie I learned freedom of will and
    undeviating steadiness of purpose ; 
    and to look to nothing else, 
    not even for a moment,
    except to reason ; 
*****************************************

"""
import numpy as np
from scipy import stats
from scipy import optimize
import multiprocessing as mp 
from path_handler import PathHandler
import h5py 
import datetime
import os 


# global variables 
paths=PathHandler()
extra_info = "EQ6_divided_by_EQ2"
quantity_names  = ["mu", "sigma", "shape", "scale", "loc", "alpha", "HYDROUS", "ANHYDROUS"]
n_trials        = 1000


def my_model_gamma(x, mu, sigma, shape, scale, loc, alpha):
    """ 
    The model for the band depth of the Reststrahlen band.
    """
    normal_part = (1 - alpha) * stats.norm.pdf(x, mu, sigma)
    gamma_part = alpha * stats.gamma.pdf(x, shape, scale=scale, loc=loc)
    return normal_part, gamma_part


def log_likelihood_gamma(params, data):
    mu, sigma, shape, scale, loc, alpha = params
    normal_part, gamma_part = my_model_gamma(data, mu, sigma, shape, scale, loc, alpha)
    return -np.sum(np.log(normal_part + gamma_part))


def constrain_guess_gauss_gamma(data):
    numeric_threshold = 1e-10
    
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    shape_init,scale_init,loc_init=stats.gamma.fit(data)
    alpha_init = 0.5
    
    mu_bounds,sigma_bounds                  = (numeric_threshold, np.inf), (0, np.inf)
    shape_bounds,scale_bounds,loc_bounds    = (numeric_threshold, np.inf), (numeric_threshold, np.inf), (numeric_threshold, np.inf)
    alpha_bounds                            = (0, 1)

    if not (mu_bounds[0] < mu_init) and (mu_init < mu_bounds[1]):
        print("mu_init out of bounds")
        print("mu_init", mu_init)
        print("mu_bounds", mu_bounds)
    if not (sigma_bounds[0] < sigma_init) and (sigma_init < sigma_bounds[1]):
        print("sigma_init out of bounds")
        print("sigma_init", sigma_init)
        print("sigma_bounds", sigma_bounds)
    if not (shape_bounds[0] < shape_init) and (shape_init < shape_bounds[1]):
        print("shape_init out of bounds")
        print("shape_init", shape_init)
        print("shape_bounds", shape_bounds)
    if not (scale_bounds[0] < scale_init) and (scale_init < scale_bounds[1]):
        print("scale_init out of bounds")
        print("scale_init", scale_init)
        print("scale_bounds", scale_bounds)
    if not (loc_bounds[0] < loc_init) and (loc_init < loc_bounds[1]):
        print("loc_init out of bounds")
        print("loc_init", loc_init)
        print("loc_bounds", loc_bounds)
    if not (alpha_bounds[0] < alpha_init) and (alpha_init < alpha_bounds[1]):
        print("alpha_init out of bounds")
        print("alpha_init", alpha_init)
        print("alpha_bounds", alpha_bounds)
    # put them together
    bounds = [mu_bounds, sigma_bounds, shape_bounds, scale_bounds, loc_bounds, alpha_bounds]
    initial_guess = [mu_init, sigma_init, shape_init, scale_init, loc_init, alpha_init]
    return initial_guess,bounds


def model_switcher(survey_name):
    models={
        "EQ2":"one_gauss",
        "EQ6":"one_gauss"}
    return models[survey_name]


def open_EQ2_EQ6_band_depths():
    survey_name="EQ2"
    model_name=model_switcher(survey_name)
    inname = paths.band_depths_on_shape_model_fname(survey_name, model_name)
    with h5py.File(inname, 'r') as file:
        EQ2 = file['band_depths'][:,:]
    survey_name="EQ6"
    model_name=model_switcher(survey_name)
    inname = paths.band_depths_on_shape_model_fname(survey_name, model_name)
    with h5py.File(inname, 'r') as file:
        EQ6 = file['band_depths'][:,:]
    return EQ2,EQ6


def worker(i,ratio):
    """
    i is the trial number or monte-carlo index
    ratio is the band depth ratio for all facets of trial i
    """
    try:
        outpath = paths.temp_global_band_depth_ratio(extra_info)
        outfilename = outpath + f"/trial_{i}.txt"
        init_guess, bounds = constrain_guess_gauss_gamma(ratio)
        results = optimize.minimize(log_likelihood_gamma, init_guess, args=(ratio,), bounds=bounds, method='Nelder-Mead')
        mu, sigma, shape, scale, loc, alpha = results.x
        ## INTEGRATE ##
        HYDROUS = alpha * stats.gamma.cdf(1, shape, scale=scale, loc=loc)
        ANHYDROUS = alpha * stats.gamma.sf(1, shape, scale=scale, loc=loc)        
        outdata = np.array([mu, sigma, shape, scale, loc, alpha, HYDROUS, ANHYDROUS])
        outdata.tofile(outfilename, sep=",")
        return None
    except Exception as e:
        print(f"Error in trial {i}: {e}")
        return None



def combine_temp_files():
    outarray        = np.zeros((n_trials, len(quantity_names)))
    inpath          = paths.temp_global_band_depth_ratio(extra_info)
    for i in range(n_trials):
        tempfilename = inpath + f"/trial_{i}.txt"
        if os.path.exists(tempfilename):
            mu, sigma, shape, scale, loc, alpha, HYDROUS, ANHYDROUS = np.loadtxt(tempfilename, delimiter=",", unpack=True)
            outarray[i] = [mu, sigma, shape, scale, loc, alpha, HYDROUS, ANHYDROUS]    
        else:
            print(f"File {tempfilename} does not exist")
    return outarray


def save_global_band_depth_ratio(outarray):
    filename=paths.global_distributions("EQ6_EQ2", "Band_depth_ratio_gauss_gamma.h5")
    with h5py.File(filename, "w") as f:
        f.attrs["description"] = "This file contains the fits to the band depth ratio distributions for EQ2_EQ6"
        f.attrs["date_created"] = str(datetime.datetime.now())
        f.attrs["created_by"] = "Salvatore Ferrone"
        f.create_dataset("quantity_names", data=np.array(quantity_names, dtype='S'))
        f.create_dataset("fit_parameters", data=outarray)
    return None


def main():
    
    EQ2,EQ6 = open_EQ2_EQ6_band_depths()
    n_facets,n_trials = EQ2.shape
    
    ## prepare data for parallel loop without NaNs
    tasks = []

    for ii in range(n_trials):   
        ratio = EQ6[:,ii]/EQ2[:,ii]
        ratio = ratio[np.isfinite(ratio)]
        tasks.append((ii,ratio))
        
        
    starttime=datetime.datetime.now()
    ncpu = mp.cpu_count()
    with mp.Pool(ncpu) as pool:
        pool.starmap(worker, tasks)
    endtime=datetime.datetime.now()
    print("Computation time", endtime-starttime)
    
    outarray = combine_temp_files()
    
    save_global_band_depth_ratio(outarray)




if __name__=="__main__":
    main()