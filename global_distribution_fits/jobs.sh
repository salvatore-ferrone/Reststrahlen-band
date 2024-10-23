#!/bin/bash
#SBATCH --job-name=Global_distribution_fit
#SBATCH --output=/dev/null
#SBATCH --error=job_%A_%a.err
#SBATCH --time=1339
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=medium
#SBATCH --mail-user=salvatore.ferrone@obspm.fr


module purge 

# Activate the Python virtual environment
source /obs/sferrone/Reststrahlen-band/pymc_pkg/bin/activate


python perform_all_fits.py