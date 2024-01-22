#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --ntasks=2
#SBATCH --error=error_file_jobsh.txt
#SBATCH --job-name=python_cpu
#SBATCH --mem=40G
#SBATCH --output=output.log

module purge
module load Python/3.9.6-GCCcore-11.2.0


# Check if at least one argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <problem> [<horizon>]"
    exit 1
fi

source /scratch/s3918343/venvs/thesis/bin/activate
pip install -r requirements.txt
echo "Argument 1: $1"


cd /scratch/s3918343/venvs/thesis/Thesis

python experiment.py problem=$1 horizon=$2 iter=5
echo "DONE"

deactivate
