#!/bin/bash
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --error=output.log
#SBATCH --output=output.log
## SBATCH --time=5-00:00:00

module load cuda/10.1

source /apps2/anaconda/5.1.0/etc/profile.d/conda.sh

conda activate gurobi

GRB_LICENSE_FILE=/home/bbg16101/PerturbPaths/gurobi.lic python run_experiments.py