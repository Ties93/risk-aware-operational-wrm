#!/bin/bash
#SBATCH -J optimization_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:0:00
#SBATCH --partition=genoa
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.log
echo starting job
module load 2022
module load Miniconda3/4.12.0
source /gpfs/admin/_hpc/sw/arch/AMD-ZEN2/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate optimization
method='energy'
var_wl=-0.4
max_threads=32
start_from_scratch=False
p_chance=1
wl_constraint_type='cvar'
n_scenarios=5
cvar_wl=-0.3
cvar_alpha=0.99
tree_based=False
year=2019
h_max=-0.3
export OMP_NUM_THREADS=$max_threads
set -euo pipefail
python 'optimization/src/run_sim_date.py' n_scenarios=$n_scenarios distance_metric=$method wl_constraint_type=$wl_constraint_type var_wl=$var_wl cvar_wl=$cvar_wl cvar_alpha=$cvar_alpha p_chance=$p_chance tree_based=$tree_based max_threads=$max_threads start_from_scratch=$start_from_scratch h_max=$h_max year=$year