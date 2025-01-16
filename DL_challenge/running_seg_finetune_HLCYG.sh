#!/bin/bash

#SBATCH --account phwq4930-renal-canc
#SBATCH --qos epsrc
#SBATCH --time 0-00:30:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 12
#SBATCH --mem 200G
#SBATCH --constraint a100_80

set -x

module purge; module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3
conda init bash
source /bask/homes/r/ropj6012/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/bask/projects/p/phwq4930-renal-canc/conda_env/ovseg_env/lib


# Location of conda environment
CONDA_ENV_PATH="/bask/projects/p/phwq4930-renal-canc/conda_env/ovseg_env"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.08,0.06,0.04,0.02,0.01]

# Run job
python /bask/homes/r/ropj6012/KCD/KCD/Segmentation/HPC_Scripts/HLCYG/finetune_coregv3noised_noisemt.py 0.01