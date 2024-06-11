#!/bin/sh 
#SBATCH --job-name=average_spec
#SBATCH --output=what_%A_%a.out
#SBATCH --error=what_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=30
#SBATCH --mail-user=salvatore.ferrone@obspm.fr
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=10




conda activate pymc5

python3 average_facet_spectra.py

