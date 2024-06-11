import h5py 
from path_handler import PathHandler #type: ignore
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys


plt.style.use('dark_background')

def main(i):

    my_facet="F"+str(i)
    outpath="/obs/sferrone/Reststrahlen-band/quickplots/"
    outname=my_facet+"-"+"banddepth"+".png"
    
    ratio1,ratio3,ratio4,ratio6=obtain_ratios(my_facet)
    times_str=create_time_stamps()
    out_array=store_ratios(ratio1,ratio3,ratio4,ratio6)
    
    ## do the plot 
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for ii in range(ratio1.shape[0]):
        ax.plot(times_str, out_array[ii,:],alpha=0.01,color='white')
    ax.set_ylabel("Band Depth Ratios")
    ax.set_xlabel("Phase angle [time hh:mm]")
    ax.set_title(my_facet)
    ax.set_ylim(0.3,1.5)
    
    fig.savefig(outpath+outname,dpi=300)
    plt.close(fig)
    print("Saved",outpath+outname)
    
    
def store_ratios(ratio1,ratio3,ratio4,ratio6):
    out_array = np.zeros((ratio1.shape[0],5))
    for ii in range(ratio1.shape[0]):
        out_array[ii,0] = 1
        out_array[ii,1] = ratio4[ii]
        out_array[ii,2] = ratio3[ii]
        out_array[ii,3] = ratio1[ii]
        out_array[ii,4] = ratio6[ii]
    return out_array

def create_time_stamps():
    # Create a datetime object
    T1 = datetime.time(15, 0, 0)
    T2 = datetime.time(3, 20, 0)
    T3 = datetime.time(12, 30, 0)
    T4 = datetime.time(10, 00, 0)
    T6 = datetime.time(20, 40, 0)
    times = np.concatenate(([T2], [T4], [T3], [T1], [T6]))
    times_str = [t.strftime('%H:%M') for t in times] 
    return times_str

def obtain_ratios(my_facet):
    paths=PathHandler()
    EQ1=h5py.File(paths.facet_spectra("EQ1",my_facet),'r')
    EQ2=h5py.File(paths.facet_spectra("EQ2",my_facet),'r')
    EQ3=h5py.File(paths.facet_spectra("EQ3",my_facet),'r')
    EQ4=h5py.File(paths.facet_spectra("EQ4",my_facet),'r')
    EQ6=h5py.File(paths.facet_spectra("EQ6",my_facet),'r')
    
    
    ratio1=EQ1['band_depth'][:]/EQ2['band_depth'][:]
    ratio3=EQ3['band_depth'][:]/EQ2['band_depth'][:]
    ratio4=EQ4['band_depth'][:]/EQ2['band_depth'][:]
    ratio6=EQ6['band_depth'][:]/EQ2['band_depth'][:]
    return ratio1,ratio3,ratio4,ratio6


if __name__=="__main__":
    i=int(sys.argv[1])
    main(i)