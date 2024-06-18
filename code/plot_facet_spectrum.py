import numpy as np #type: ignore
import sys 
import matplotlib.pyplot as plt #type: ignore
plt.style.use('dark_background')
import h5py #type: ignore
sys.path.append("../code/")
from path_handler import PathHandler #type: ignore
import fit_one_spectrum as FOS #type: ignore






def main(survey_name,model_name,facet_number):
    
    assert isinstance(facet_number,str), "Facet number must be a string"
    
    depth_lims  =   (0.01,0.035)
    ylimits     =   (0.95,1.01)
    outname = "../quickplots/"+facet_number+"_"+survey_name+".png"
    
    paths       =   PathHandler()
    path_to_spectra=paths.otes_csv(survey_name)
    path_to_wave_numbers=paths.wavenumbers
    spectra_csv, wave_nb=FOS.open_data(path_to_spectra,path_to_wave_numbers)
    
    wv_num_min, wv_num_max, _,_,_=FOS.set_parameters()
    
    fitfile=h5py.File(paths.facet_spectra(survey_name,facet_number),'r')
    
    #### Get the fits
    wavenumber, mindexes, facet_mean_spectra, facet_mean_continuum, band_depth=extract_fits(fitfile)
    #### get the raw spectra
    raw_=extract_raw(fitfile,spectra_csv,wave_nb,wv_num_min,wv_num_max)
    
    
    fig, _=do_plotting(wavenumber,raw_,facet_mean_spectra,facet_mean_continuum,mindexes,band_depth,depth_lims,ylimits)
    fig.suptitle("{:s} {:s}".format(survey_name,facet_number))
    fig.tight_layout()
    fig.savefig(outname,dpi=300)

def do_plotting(wavenumber,raw_,facet_mean_spectra,facet_mean_continuum,mindexes,band_depth,depth_lims,ylims):
    
    numfits = 1000
    nbins = int(np.ceil(np.sqrt(numfits)))
    
    axis0, axis1, linespecs, depthspecs, scatterspecs, meanlinespecs = properties(depth_lims,ylims)
    mean_raw=np.mean(raw_,axis=0)
    fig,axis=plt.subplots(1,2,figsize=(10,3))
    for i in range(facet_mean_spectra.shape[0]):
        xbands=[wavenumber[mindexes[i]],wavenumber[mindexes[i]]]
        ybands=[facet_mean_spectra[i][mindexes[i]],facet_mean_continuum[i][mindexes[i]]]
        axis[0].plot(wavenumber,facet_mean_spectra[i],**linespecs)
        axis[0].plot(wavenumber,facet_mean_continuum[i],**linespecs)
        axis[0].plot(xbands,ybands,**depthspecs)
    
    for ii in range(raw_.shape[0]):
        axis[0].scatter(wavenumber,raw_[ii],**scatterspecs)
    axis[0].plot(wavenumber,mean_raw,**meanlinespecs)
    axis[1].hist(band_depth,bins=nbins,color="w",alpha=0.5);
    axis0['ylim']=ylims
    axis[0].set(**axis0)
    axis[1].set(**axis1)
    fig.tight_layout()
    return fig, axis



def properties(
    depth_lims=(0.01,0.035),
    ylims=(0.95,1.01),
    axis0={"xlabel":"Wavenumber (cm$^{-1}$)","ylabel":"Reflectance"},
    axis1={"xlabel":"Band depth"},
    linespecs={"color":"white","linewidth":1,"alpha":0.01},
    depthspecs={"color":"red","linewidth":1,"alpha":0.01},
    scatterspecs={"color":"blue","s":1,"alpha":0.5},
    meanlinespecs={"color":"blue","linewidth":1,"alpha":1},
):
    axis0['ylim']=ylims
    axis1['xlim']=depth_lims
    return  axis0, axis1, linespecs, depthspecs, scatterspecs, meanlinespecs

    

    
def extract_raw(fitfile,spectra_csv,wave_nb,wv_num_min,wv_num_max):
    wavenumber, spectra_filter=FOS.filter_by_wavenumber(wave_nb,wv_num_min,wv_num_max)
    nsclks = len(fitfile['sclks'])
    raw_ = np.zeros((nsclks, len(wavenumber)))
    for ii in range(nsclks):
        row_index=np.where(spectra_csv['sclk_string']== fitfile['sclks'][:][ii].decode("utf-8"))[0][0]
        my_spectrum=spectra_csv.iloc[row_index][spectra_filter].to_numpy()
        raw_[ii]=np.array(my_spectrum,dtype=float)
    return raw_
    
    
def extract_fits(myfile):
    wavenumber          =   myfile["wavenumber"][:]
    mindexes            =   myfile["mindexes"][:]
    facet_mean_spectra  =   myfile['facet_mean_spectra'][:]
    facet_mean_continuum=   myfile['facet_mean_continuum'][:]
    band_depth          =   myfile['band_depth'][:] 
    return wavenumber, mindexes, facet_mean_spectra, facet_mean_continuum, band_depth
    
    
if __name__=="__main__":
    survey_name="EQ2"
    model_name="one_gauss"
    facet_number="F10"
    main(survey_name,model_name,facet_number)