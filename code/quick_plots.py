import h5py 
import numpy as np
import matplotlib.pyplot as plt
import perform_bayes_fit as PBF
import fit_one_spectrum as FOS
import sys 



def check_fit(survey_name, row_index):
    # wv_num_min, wv_num_max, _,_,_=FOS.set_parameters()
    # path_to_spectra, path_to_wave_numbers=FOS.get_paths(survey_name=survey_name)
    # spectra_csv, wave_nb=FOS.open_data(path_to_spectra, path_to_wave_numbers)
    # wavenumber, spectra_filter=FOS.filter_by_wavenumber(wave_nb, wv_num_min, wv_num_max)
    
    # sclk=spectra_csv.iloc[row_index]['sclk_string']
    # my_spectrum=spectra_csv.iloc[row_index][spectra_filter].to_numpy()
    # my_spectrum=np.array(my_spectrum,dtype=float)   
    
    # get the fit
    path_to_fit= "/scratch2/sferrone/OREX/bayes_fits/"+survey_name+"/"+survey_name+"_OTES_"+str(row_index).zfill(4)+".h5"
    
    with h5py.File(path_to_fit,'r') as fp:
        fit_array=fp['chain'][:]
        spectrum=fp['chain'].attrs['spectrum'][:]
        wavenumber=fp['chain'].attrs['wavenumber'][:]
        sclk=fp['chain'].attrs['sclk']
        
    plt.style.use('dark_background') 
    # increase the fontsize
    plt.rcParams.update({'font.size': 16})
    
    fig, axis=plt.subplots(1,1,figsize=(8,5))
    fig, axis=add_spectrum(fig,axis,wavenumber,spectrum)
    fig, axis=add_plot_fits(fig,axis,fit_array, wavenumber)
    
    axis.set_title("{:s} {:s}".format(survey_name, sclk))

    axis.set_xlabel("Wavenumber [cm$^{-1}$]")  
    axis.set_ylabel("Emisivity ")
    
    axis.set_ylim(0.95,1.01)
    
    fig.tight_layout()


    fname = "{:s}_OTES_{:04d}.png".format(survey_name, row_index)
    fig.savefig("/obs/sferrone/Reststrahlen-band/quickplots/"+fname, dpi=300)
    plt.close(fig)

def add_spectrum(fig,axis,wavenumber,spectrum):
    properties = {"color":"blue","label":"Spectrum","s":5,"marker":"o"}
    axis.scatter(wavenumber,spectrum,**properties)
    return fig, axis

def add_plot_fits(fig,axis,fit_array, wavenumber):
    
    params={"alpha":0.01,"color":"white"}
    for i in range(fit_array.shape[1]):
        m=fit_array[0,i]
        b=fit_array[1,i]
        A0=fit_array[2,i]
        A1=fit_array[3,i]
        k0=fit_array[4,i]
        k1=fit_array[5,i]
        sigma0=fit_array[6,i]
        sigma1=fit_array[7,i]
        spectrum=PBF.model_spectrum(wavenumber, b, m, [A0, A1], [k0, k1], [sigma0, sigma1])
        continuum=PBF.continuum(wavenumber, b, m)
        axis.plot(wavenumber, spectrum, **params)
        axis.plot(wavenumber, continuum, **params)
        

    return fig, axis

if __name__=="__main__":
    row_index = int(sys.argv[1])
    check_fit("EQ2", row_index)