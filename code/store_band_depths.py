"""
After 
    1.  All of the bayesian fits are computed and stored
    2.  All of the facet averages have been completed 
    2.b the band depth measurements have been completed and saved
We will now store the band depth measurements in a single file per facet.
"""


import numpy as np 
import h5py
import os 
from concurrent.futures import ProcessPoolExecutor
from astropy.io import fits
import sys 
from path_handler import PathHandler
from astropy import units as u
import datetime


def read_hdf5(file_name,n_trials):
    """Function to read a single HDF5 file and return the 'band_depth' data."""
    # check if file exists first
    if os.path.isfile(file_name) == False:
        return np.nan*np.zeros(n_trials)
    else:
        with h5py.File(file_name, 'r') as file:
            return file['band_depth'][:]



def model_switcher(survey_name):
    models={
        "EQ2":"one_gauss",
        "EQ6":"one_gauss"
    }
    return models[survey_name]
    
    
def write_output(outname, survey_name, banddepths, model_name):


        n_facets = banddepths.shape[0]
        n_trials = banddepths.shape[1]
        facet_array = ["F"+str(i) for i in range(1,n_facets+1)]
        monte_carlo_indexes = np.arange(n_trials)
        with  h5py.File(outname, 'w') as myoutfile:
            myoutfile.create_dataset('band_depths', data=banddepths)
            myoutfile.create_dataset('facet_array', data=facet_array)
            myoutfile.create_dataset('monte_carlo_indexes', data=monte_carlo_indexes)
            myoutfile.attrs["model_name"] = model_name
            myoutfile.attrs["survey_name"] = survey_name
            myoutfile.attrs['author'] = "Salvatore Ferrone"
            myoutfile.attrs['email'] = "salvatore.ferrone@uniroma1.it"
            myoutfile.attrs['date'] = str(datetime.datetime.now())
        print("output saved to", outname)
            
            
def main(survey_name, n_trials = 1000):
    
    paths=PathHandler()
    model_name = model_switcher(survey_name)
    outname = paths.band_depths_on_shape_model_fname(survey_name, model_name)
    if os.path.isfile(outname):
        print(outname, "output file already exists\n NO COMPUTE")
        return

    with fits.open("/scratch2/sferrone/OREX/shape_models/g_06310mm_spc_tes_0000n00000_v020.fits" , mode='readonly') as shapemodel:
        n_facets = shapemodel[1].data.shape[0]
    # estimate the size of the output array
    outsize=n_facets*n_trials* 64 * u.bit
    print("size out output array", outsize.to(u.Mbyte))
    
    # PREALLOCATE OUTPUT ARRAY
    banddepths = np.zeros((n_facets,n_trials))
    filenames   =   [paths.facet_spectra(survey_name, model_name, "F"+str(i)) for i in range(1,n_facets+1)]    
    
    # extract the data
    starttime=datetime.datetime.now()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(read_hdf5, filenames,n_facets*[n_trials]))
    endtime  = datetime.datetime.now()
    print("Time for Extraction", endtime-starttime)   
    
    # store the data
    for i in range(n_facets):
        banddepths[i,:] = results[i]
        
    # save the output
    write_output(outname, survey_name, banddepths, model_name)


if __name__=="__main__":
    survey_name = sys.argv[1]
    main(survey_name)