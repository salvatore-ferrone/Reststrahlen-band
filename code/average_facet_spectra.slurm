#!/bin/sh 
#SBATCH --job-name=average_spec
#SBATCH --output=what_%A_%a.out
#SBATCH --error=what_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=300
#SBATCH --mail-user=salvatore.ferrone@obspm.fr
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=20




conda activate pymc5

python3 average_facet_spectra.py

