#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=python_cpu
#SBATCH --mem=8000
module purge
module load Python/3.9.6-GCCcore-11.2.0

 
source /scratch/s3918343/venvs/thesis/bin/activate

cd /scratch/s3918343/venvs/thesis/Thesis

python experiment.py problem=dectiger horizon=3 iter=3

deactivate
