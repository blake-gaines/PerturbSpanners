#!/bin/bash
##SBATCH --account=jib10001 --partition=GpuPriorityJB --qos=jib10001gpu
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:0
##SBATCH --exclude=gpu[01-08],gpu[10],gtx[01-16]
#SBATCH --error=output.log
#SBATCH --output=output.log
## SBATCH --time=5-00:00:00
##SBATCH --mail-user=blake@uconn.edu
##SBATCH --mail-type=BEGIN

module load cuda/10.1

source /apps2/anaconda/5.1.0/etc/profile.d/conda.sh

conda activate gurobi

GRB_LICENSE_FILE=/home/bbg16101/PerturbPaths/gurobi.lic python run_experiments.py