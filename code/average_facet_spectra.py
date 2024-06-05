import h5py #type: ignore
import json
import os 
import numpy as np 
from path_handler import PathHandler #type: ignore
import fit_one_spectrum as FOS #type: ignore
import perform_bayes_fit as PBF #type: ignore
import multiprocessing as mp 
import datetime


def main(survey_name="EQ3",ncpu=10):
    paths=PathHandler()
    spectra_csv, wavenumbers = FOS.open_data(paths.otes_csv("EQ3"), paths.wavenumbers)
    
    with open(paths.json_getspots(survey_name)) as fp:
        getspots = json.load(fp)
    facetstrs=list(getspots.keys())
    
    ### LOOP OVER ALL FACETS ###
    Nfacets = len(facetstrs)
    # for jj in range(10):
    start_time = datetime.datetime.now()
    pool=mp.Pool(ncpu)
    for jj in range(150,Nfacets):
        pool.apply_async(loop_over_facets,args=(jj,getspots,facetstrs,spectra_csv,survey_name,paths))
    pool.close()    
    pool.join()
    # for jj in range(40,90):
        # loop_over_facets(jj,getspots,facetstrs,spectra_csv,survey_name,paths)
    end_time = datetime.datetime.now()
    print("Time elapsed: ",end_time-start_time)
    print("Done")
    
    
def loop_over_facets(jj,getspots,facetstrs,spectra_csv,survey_name,paths):
    facet_str= facetstrs[jj]
    present_sclks,fnames=get_present_sclks_and_bayes_filenames_for_facet(\
        jj,getspots,spectra_csv,survey_name,paths)
    facet_mean_spectra,facet_mean_continuum,wavenumber =mean_spectra_and_band_depth(\
        fnames)
    mindexes,band_depth=measure_band_depth(\
        facet_mean_spectra,facet_mean_continuum)
    write_out(\
        paths,survey_name,facet_str,facet_mean_spectra,facet_mean_continuum,wavenumber,mindexes,band_depth,present_sclks,fnames)
    

def make_spectrum_and_continuum(i,wavenumber,fit_array):
    m=fit_array[0,i]
    b=fit_array[1,i]
    A0=fit_array[2,i]
    A1=fit_array[3,i]
    k0=fit_array[4,i]
    k1=fit_array[5,i]
    sigma0=fit_array[6,i]
    sigma1=fit_array[7,i]
    spectrum=PBF.model_spectrum(wavenumber, b, m, [A0, A1], [k0, k1], [sigma0, sigma1])
    continuum = m*wavenumber+b
    return spectrum,continuum


def snag_fit_data(fname):
    with h5py.File(fname,'r') as fp:
        fit_array=fp['chain'][:]
        spectrum=fp['chain'].attrs['spectrum'][:]
        wavenumber=fp['chain'].attrs['wavenumber'][:]
        sclk=fp['chain'].attrs['sclk']
    return fit_array, spectrum, wavenumber, sclk


def get_present_sclks_and_bayes_filenames_for_facet(jj,getspots,spectra_csv,survey_name,paths):
    facet_str = "F"+str(jj+1)
    n_sclk = len(getspots[facet_str]['sclks'])
    fnames,present_sclks = [],[]
    for ii in range(n_sclk):
        sclk=getspots[facet_str]['sclks'][ii]
        subset=spectra_csv.loc[spectra_csv['sclk_string']==sclk]
        if subset.empty:
            print("Beware ",sclk, "not in spectra CSV", survey_name)
            continue
        else:
            indx = subset.index[0]
            fname=paths.bayes_fits_fname(survey_name,indx)
            if os.path.exists(fname):
                fnames.append(fname)
                present_sclks.append(sclk)
            else:
                print(fname, "n'existe pas")
    return present_sclks,fnames

### now iterate over the present sclks
def mean_spectra_and_band_depth(fnames,):
    ## initialize the output 
    fit_array, raw_spectrum, wavenumber, sclk = snag_fit_data(fnames[0])
    n_spec, n_fits = len(fnames),fit_array.shape[1]
    spectral_array = np.zeros((n_spec,n_fits,raw_spectrum.shape[0]))
    continuum_array = np.zeros((n_spec,n_fits,raw_spectrum.shape[0]))
    # iterate over each sclk
    for ii in range(n_spec):
        fit_array, raw_spectrum, wavenumber, sclk = snag_fit_data(fnames[ii])
        # iterate over each bayes fit 
        for jj in range(n_fits):
            temp_spec,temp_continu = make_spectrum_and_continuum(jj,wavenumber,fit_array)
            spectral_array[ii,jj,:] = temp_spec
            continuum_array[ii,jj,:] = temp_continu
    # average down
    facet_mean_spectra=np.mean(spectral_array,axis=0)
    facet_mean_continuum=np.mean(continuum_array,axis=0)
    return facet_mean_spectra,facet_mean_continuum,wavenumber


def measure_band_depth(facet_mean_spectra,facet_mean_continuum):
    n_fits = facet_mean_spectra.shape[0]
    mindexes = np.zeros(n_fits,dtype=int)
    band_depth = np.zeros(n_fits)
    for ii in range(n_fits):
        mindexes[ii] = np.argmin(facet_mean_spectra[ii,:])
        band_depth[ii] = facet_mean_continuum[ii,mindexes[ii]]-facet_mean_spectra[ii,mindexes[ii]]
    return mindexes,band_depth


def write_out(paths,survey_name,facet_str,facet_mean_spectra,facet_mean_continuum,wavenumber,mindexes,band_depth,present_sclks,fnames):
    filename=paths.facet_spectra(survey_name,facet_str)
    with h5py.File(filename,'w') as myfile:
        myfile.create_dataset("facet_mean_spectra", data=facet_mean_spectra)
        myfile.create_dataset("facet_mean_continuum", data=facet_mean_continuum)
        myfile.create_dataset("wavenumber", data=wavenumber)
        myfile.create_dataset("mindexes", data=mindexes)
        myfile.create_dataset("band_depth", data=band_depth)
        myfile.create_dataset("sclks", data=present_sclks)
        myfile.create_dataset("fnames", data=np.array(fnames,dtype='S'))

        myfile.attrs["author"]="salvatore ferrone"
        myfile.attrs["email"]="salvatore.ferrone@uniroma1.it"
        myfile.attrs["survey_name"]=survey_name
        myfile.attrs["facet_str"]=facet_str
        myfile.attrs["creation_date"]=str(np.datetime_as_string(np.datetime64('today'), unit='D'))
    print("Wrote ",filename)


if __name__=="__main__":
    main()