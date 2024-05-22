import pymc
import numpy as np 
import pandas as pd
import h5py 
import perform_bayes_fit as PBF
import sys 
import datetime

def set_parameters(wv_num_min=520,wv_num_max=1150,chain_length = 1000,chain_sample=250,n_chains=4):
    return wv_num_min, wv_num_max, chain_length,chain_sample,n_chains

def get_paths(survey_name="EQ3"):
    path_to_spectra="/scratch2/sferrone/OREX/OTES/"+survey_name+"_data_spectra.csv"
    path_to_wave_numbers="/scratch2/sferrone/OREX/OTES/OTES_wnb.csv"
    return path_to_spectra, path_to_wave_numbers

def open_data(path_to_spectra,path_to_wave_numbers):
    spectra_csv = pd.read_csv(path_to_spectra)
    wave_nb=pd.read_csv(path_to_wave_numbers).to_numpy()[0]
    return spectra_csv, wave_nb

def filter_by_wavenumber(wave_nb,wv_num_min,wv_num_max):
    cond1=(wave_nb>=wv_num_min) 
    cond2=(wave_nb<=wv_num_max)
    cond=cond1 & cond2
    wavenumber=wave_nb[cond]
    spectra_filter=np.insert(cond,0,False) # drop the sclk column
    return wavenumber, spectra_filter

def get_variable_names():
    return ["m","b","A_0","A_1","kbar_0","kbar_1","sigma_0","sigma_1"]

def write_to_hdf5(
    survey_name,
    row_index,
    outarray,
    variables,
    sclk,
    n_chains,
    chain_sample,
    chain_length,
    wavenumber,
    my_spectrum,
    wv_num_min,
    wv_num_max,
    path_to_spectra,
    path_to_wave_numbers
    
):
    outpath="/scratch2/sferrone/OREX/fits/"+survey_name+"_OTES_"+str(row_index).zfill(4)+".h5"
    with h5py.File(outpath,'w') as f:
        dset=f.create_dataset("chain",data=outarray)
        dset.attrs['variables']=variables
        dset.attrs['sclk']=sclk
        dset.attrs['n_chains']=n_chains
        dset.attrs['chain_sample']=chain_sample
        dset.attrs['chain_length']=chain_length
        dset.attrs['wavenumber']=wavenumber
        dset.attrs['spectrum']=my_spectrum
        dset.attrs['sclk']=sclk
        dset.attrs['survey_name']=survey_name
        dset.attrs['wv_num_min']=wv_num_min
        dset.attrs['wv_num_max']=wv_num_max
        dset.attrs['path_to_spectra']=path_to_spectra
        dset.attrs['path_to_wave_numbers']=path_to_wave_numbers
        dset.attrs['row_index']=row_index

def main(row_index,survey_name):
    
    start_time = datetime.datetime.now()
    wv_num_min, wv_num_max, chain_length,chain_sample,n_chains=\
        set_parameters()
    path_to_spectra, path_to_wave_numbers=\
        get_paths(survey_name)
    spectra_csv, wave_nb=\
        open_data(path_to_spectra,path_to_wave_numbers)
    wavenumber, spectra_filter=\
        filter_by_wavenumber(wave_nb,wv_num_min,wv_num_max)
    variable_names=get_variable_names()
    
    sclk=spectra_csv.iloc[row_index]['sclk_string']
    my_spectrum=spectra_csv.iloc[row_index][spectra_filter].to_numpy()
    my_spectrum=np.array(my_spectrum,dtype=float)
    
    print("Data loaded",datetime.datetime.now()-start_time)
    # build the model 
    model=PBF.build_pymc_model(wavenumber,my_spectrum)
    print("Model built\n Doing Trace")
    
    with model:
        trace=pymc.sample(chain_length)
    # extract the trace
    print("Trace done\n Storing",datetime.datetime.now()-start_time)
    trace_dict=PBF.store_trace_in_dict(sclk,variable_names,trace,chain_length,chain_sample)
    
    # store in output array
    outarray=np.zeros((len(variable_names),chain_sample*n_chains))
    for i in range(len(variable_names)):
        outarray[i,:]=trace_dict[sclk][variable_names[i]]
    
    write_to_hdf5(
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
        path_to_wave_numbers
    )
    print("Done",datetime.datetime.now()-start_time)


if __name__=="__main__":
    survey_name="EQ3"
    row_index=int(sys.argv[1])
    main(row_index,survey_name)