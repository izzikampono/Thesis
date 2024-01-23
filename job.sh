#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --error=error_file_jobsh.txt
#SBATCH --job-name=$1
#SBATCH --mem=5G
#SBATCH --output=output.log

module purge
module load Python/3.9.6-GCCcore-11.2.0


# Check if at least one argument is provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <problem> [<horizon>] [num_iter]"
    exit 1
fi
source /scratch/s3918343/venvs/thesis/bin/activate
echo : "initialized python evironment"
module load CPLEX/22.1.1-GCCcore-11.2.0
cplex -c set parallel -1
cplex quit
cplex -c set threads 0
cplex quit
echo : "\n\n\n Loaded Cplex and set to parallel computing \n\n\n"

echo "Run problem : $1 with horizon: $2 and iter : $3"


cd /scratch/s3918343/venvs/thesis/Thesis

python experiment_server.py problem=$1 horizon=$2 iter=$3
echo " SOLVING DONE"

deactivate
