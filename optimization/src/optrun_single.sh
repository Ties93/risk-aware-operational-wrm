#!/bin/bash
#SBATCH -J jupyter_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --partition=genoa
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.log
module load 2022
module load Miniconda3/4.12.0
source /gpfs/admin/_hpc/sw/arch/AMD-ZEN2/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate optimization
n_scenarios=2
method='energy'
wl_constraint_type='cvar'
var_wl=-0.4
cvar_wl=-0.3
cvar_alpha=0.99
p_chance=1
max_threads=64
start_from_scratch=False
tree_based=False
export OMP_NUM_THREADS=$max_threads
set -euo pipefail
python 'optimization/src/run_single_simulation.py' n_scenarios=$n_scenarios distance_metric=$method wl_constraint_type=$wl_constraint_type var_wl=$var_wl cvar_wl=$cvar_wl cvar_alpha=$cvar_alpha p_chance=$p_chance tree_based=$tree_based max_threads=$max_threads start_from_scratch=$start_from_scratch