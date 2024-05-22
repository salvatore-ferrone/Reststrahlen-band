#!/bin/bash
#SBATCH --job-name=EQ3-part3
#SBATCH --array=2000-2283
#SBATCH --output=/dev/null
#SBATCH --error=../jobfiles/my_job_%A_%a.err
#SBATCH --time=1339
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=medium
#SBATCH --mail-user=salvatore.ferrone@obspm.fr

# Load the necessary modules
module purge 

# Activate the Python virtual environment
source /obs/sferrone/Reststrahlen-band/pymc_pkg/bin/activate

# Run the Python script
python fit_one_spectrum.py ${SLURM_ARRAY_TASK_ID} 2> ../jobfiles/error_${SLURM_ARRAY_TASK_ID}.txt