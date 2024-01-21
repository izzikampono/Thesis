#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --ntasks=10
#SBATCH --output=output_file.csv
#SBATCH --job-name=python_cpu
#SBATCH --mem=8000
module purge
module load Python/3.9.6-GCCcore-11.2.0


source /scratch/s3918343/venvs/thesis/bin/activate
output_file="output.csv"


cd /scratch/s3918343/venvs/thesis/Thesis

python experiment.py problem=dectiger horizon=3 iter=3 > "$output_file"

deactivate
