#!/bin/sh 
#SBATCH --job-name=EQ1_average_spec
#SBATCH --output=./outs/EQ1_%A_%a.out
#SBATCH --error=./err/what_%A_%a.err
#SBATCH --partition=medium
#SBATCH --time=1399
#SBATCH --mail-user=salvatore.ferrone@obspm.fr
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=20




conda init 

conda activate pymc5

python3 average_facet_spectra.py

