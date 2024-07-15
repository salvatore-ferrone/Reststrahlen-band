import numpy as np 
import h5py
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import os 
import multiprocessing as mp

import sys 
sys.path.append("../code/")
from path_handler import PathHandler
import compute_global_band_depth_ratio as CGBD


# global variables
paths=PathHandler()
extra_info      =   "EQ2_divided_by_EQ6"
quantity_names  =   ["mu", "sigma", "shape", "scale", "loc", "alpha", "HYDROUS", "ANHYDROUS"]
n_trials        =   1000

#### PLOT PARAMETERS
cmin,cmax   =   0.5,1.5
xmin,xmax   =   0.5,1.75


def main():

    # get the fit results
    filename=paths.global_distributions("EQ2_EQ6", "Band_depth_ratio_gauss_gamma.h5")
    mu,sigma,shape,scale,loc,alpha,HYDROUS,ANHYDROUS=extract_fit_results(filename)
    lat,lon=extract_lat_lon()
    # get the band depth ratio
    EQ2,EQ6 = CGBD.open_EQ2_EQ6_band_depths()
    n_facets=EQ2.shape[0]
    ratio = np.zeros_like(EQ2.T)
    for ii in range(n_trials):
        ratio[ii,:] = EQ2[:,ii]/EQ6[:,ii]
        
    ### PREPARE THE PLOT 
    bin_centers,bin_edges,colors,mynorm,colormap=precompute_plot_params(n_facets)
    bin_width=bin_edges[1]-bin_edges[0]    
    
    # make the outname
    mydir = "/map_and_distribution_BD_ratio/"
    outpath=paths.plots() + mydir
    os.makedirs(outpath, exist_ok=True)
    


    # make the plots
    ncpu=mp.cpu_count()
    with mp.Pool(ncpu) as pool:
        pool.starmap(worker,
            [(ii,lon,lat,ratio,mu,sigma,shape,scale,loc,alpha,bin_centers,bin_edges,colors,mynorm,colormap,bin_width,outpath) for ii in range(n_trials)])
    # worker(ii,lon,lat,ratio,mu,sigma,shape,scale,loc,alpha,bin_centers,bin_edges,colors,mynorm,colormap,bin_width,outpath)
    
    
    
def worker(ii,lon,lat,ratio,mu,sigma,shape,scale,loc,alpha,bin_centers,bin_edges,colors,mynorm,colormap,bin_width,outpath):
    fname = "frame"+str(ii).zfill(4)+".png"

    # get the ratio
    my_ratio            =   ratio[ii]
    normpart,gammapart  =   CGBD.my_model_gamma(bin_centers,mu[ii],sigma[ii],shape[ii],scale[ii],loc[ii],alpha[ii],)
    # get the normalization constant
    counts,_            =   np.histogram(ratio[ii],bins=bin_edges)
    Cnst                =   np.sum(counts)*bin_width
    counts              =   counts/Cnst
    
    ## DO THE PLOT
    fig,axis,hgram=set_up_figure()
    outs=fill_in_plot(fig,axis,hgram,lon,lat,my_ratio,colormap,mynorm,bin_centers,bin_width,counts,colors,normpart,gammapart)
    fig=outs[0]
    fig.tight_layout()
    fig.savefig(outpath+fname, dpi=300)
    # print("saved",outpath+fname)
    plt.close(fig)


def extract_fit_results(filename):
    with h5py.File(filename, "r") as myfile:
        mu=myfile['fit_parameters'][:,0]
        sigma=myfile['fit_parameters'][:,1]
        shape=myfile['fit_parameters'][:,2]
        scale=myfile['fit_parameters'][:,3]
        loc=myfile['fit_parameters'][:,4]
        alpha=myfile['fit_parameters'][:,5]
        HYDROUS=myfile['fit_parameters'][:,6]
        ANHYDROUS=myfile['fit_parameters'][:,7]
    return mu,sigma,shape,scale,loc,alpha,HYDROUS,ANHYDROUS


def extract_lat_lon():
    # get the lat lon from the shape model
    shapemodel = fits.open("/scratch2/sferrone/OREX/shape_models/g_06310mm_spc_tes_0000n00000_v020.fits" , mode='readonly') 
    lat=shapemodel[1].data['LATITUDE']
    lon=shapemodel[1].data['LONGITUDE']
    lon = np.where(lon>180, lon-360, lon)
    return lat,lon    









def set_up_figure():
    fig=plt.figure(figsize=(10,3))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 0.5], wspace=0.01)
    axis=fig.add_subplot(gs[0])
    hgram = fig.add_subplot(gs[1])    
    return fig,axis,hgram

            
        
def precompute_plot_params(n_facets,xmin=0.5,xmax=1.75,cmin=0.5,cmax=1.5):
    # precompute plot stuff
    nbins = int(np.ceil(np.sqrt(n_facets)))
    bin_edges = np.linspace(xmin,xmax,nbins+1)
    mynorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    colormap = plt.cm.rainbow  # Choose a colormap
    norm_centers = mynorm(bin_centers)  # Normalize the bin centers
    colors = colormap(norm_centers)  # Get colors 
    return bin_centers,bin_edges,colors,mynorm,colormap
    
    


def fill_in_plot(fig,axis,hgram,lon,lat,my_ratio,colormap,mynorm,bin_centers,bin_width,counts,colors,normpart,gammapart,\
    hgram_params = {
    "xlim": [xmin,xmax],
    "yticks": [],
    "ylim": [0, 3],
    "xlabel": "Band depth ratio EQ2/EQ6",},
    axis_params = { "xlim": [-180, 180],"ylim": [-90, 90],"xlabel": "Long","ylabel": "Lat","xticks": np.arange(-180, 180, 30),"yticks": np.arange(-90, 90+1, 30),}):
    
    im=axis.scatter(lon,lat,s=1,c=my_ratio,cmap=colormap,norm=mynorm,alpha=0.9)
    
    for count, bin_center, color in zip(counts, bin_centers, colors):
        hgram.bar(bin_center, count, width=bin_width, color=color, align='center')
    
    
    hgram.plot(bin_centers,normpart,linestyle='--',color='black',label='Normal')
    hgram.plot(bin_centers,gammapart,linestyle=':',color='black',label='Excess')
    hgram.plot(bin_centers,normpart+gammapart,linestyle='-',color='black',label='Total')

    hgram.set(**hgram_params)
    hgram.legend()
    axis.set(**axis_params);
    return fig,axis,hgram,im



if __name__=="__main__":
    main()