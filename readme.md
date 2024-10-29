# Scope

This code is in support of our article about discussing the distribution of hydrated materials on the surface of Bennu using OTES spectra from the OSIRIS-REx global surveys. The flow of the project is the following 


## 1. Model the OTES spectrum
We only consider wavelength range xx to xx. We are interested in this band which we either model as a linear continuum minus a gaussian OR a linear continuum minus two gaussians. The number of gaussians depends on the data quality. Here is which survey used which model:
    - EQ 1 - X
    - EQ 2 
    - EQ 3 
    - EQ 4 
    - EQ 5
    - EQ 6 

The model fits were performed using *pymc5* which is an open source python package for applying a monte-carlo markov chain fit to explore the posterior distribution on the model parameters. Like this, we are able to obtain the correlations between all the model parameters. We performed 1000 fits per spectrum. 

The resultant data product from this step is N survey $\times$ N spectra $\times$ N monte carlo
so about $5 \times 2000 \times 1000$


## 2. Facet average spectra
We then take the Palmer v 20 shape model with 49k facets and average the spectra per facet. The 

## 3. 
