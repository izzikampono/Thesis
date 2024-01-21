#!/bin/bash

# This is a Bash script that dynamically sets Slurm directives and takes an additional argument

# Check if at least two arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <problem>"
    exit 1
fi

# Access command-line arguments

problem="$1"
prob = $1
# Add more variables as needed

# Slurm script filename
slurm_script="job2.sh"

# Create the Slurm script dynamically based on command-line arguments
cat > "$slurm_script" <<EOL
#!/bin/bash
#SBATCH --job-name=python_cpu
#SBATCH --output=$prob.csv
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --error=error_file.txt
#SBATCH --mem=8000
#SBATCH --time=00:10:00

echo "Problem: $problem"

# Activate your Python environment 
module --force purge
module load Python/3.9.6-GCCcore-11.2.0
source /scratch/s3918343/venvs/thesis/bin/activate

#install all required libraries
pip install -r requirements.txt

cd /scratch/s3918343/venvs/thesis/Thesis

# Your Slurm job commands go here
python experiment.py problem="$problem" horizon=10 iter=10

deactivate
EOL

# Make the Slurm script executable
chmod +x "$slurm_script"

# Submit the Slurm job
sbatch "$slurm_script"

# Exit the script
exit 0
