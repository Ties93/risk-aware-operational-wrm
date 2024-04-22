n_scenarios = [25]#[2]#, 3, 5, 25, 50, 100]
# constraint_type = "'cvar'"
vars = [-0.4]
cvars = [-0.395, -0.3, -0.2]
alphas = [0.99, 0.95, 0.9, 0.8, 0.5] #[0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
tree_based= [False]

def make_slurm_job(name, n, var, cvar, alpha, tree):
    w = open(name, 'w', newline='\n')
    w.write('#!/bin/bash\n')
    w.write('#SBATCH -J jupyter_lab\n')
    w.write('#SBATCH --nodes=1\n')
    w.write('#SBATCH --ntasks=1\n')
    n_cpu = 64
    w.write(f'#SBATCH --cpus-per-task={n_cpu}\n') # can go to 128?
    # if (n <= 2) & (tree == False):
    #     w.write('#SBATCH --time=1:00:00\n')
    # if n <= 5:
    #     w.write('#SBATCH --time=1:00:00\n')
    # elif n <= 25:
    #     w.write('#SBATCH --time=24:00:00\n')
    # else:
    #     w.write('#SBATCH --time=48:00:00\n')
    w.write('#SBATCH --time=48:00:00\n')
    # w.write('#SBATCH --partition=genoa\n')
    w.write('#SBATCH --partition=fat\n')
    w.write('#SBATCH --output=slurm_%j.log\n')
    w.write('#SBATCH --error=slurm_%j.log\n')

    w.write('module load 2022\n')
    w.write('module load Miniconda3/4.12.0\n')
    # w.write('source /sw/arch/Centos8/EB_production/2021/software/Miniconda3/4.9.2/etc/profile.d/conda.sh\n')
    # w.write('conda init bash\n')
    w.write('source /gpfs/admin/_hpc/sw/arch/AMD-ZEN2/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh\n')
    w.write('conda activate optimization\n')

    w.write('n_scenarios=' + str(n) + '\n')
    w.write("method='energy'\n")
    w.write(f"wl_constraint_type='cvar'\n")
    w.write(f"var_wl={var}\n")
    w.write(f"cvar_wl={cvar}\n")
    w.write(f"cvar_alpha={alpha}\n")
    w.write(f"p_chance=1\n")
    w.write(f'max_threads={n_cpu}\n')
    w.write('start_from_scratch=False\n')
    w.write(f'tree_based={tree}\n')

    w.write(f'export OMP_NUM_THREADS=$max_threads\n')
    w.write('set -euo pipefail\n')
    
    # w.write("cd ..\n")
    # w.write("cd ..\n")

    w.write("python 'optimization/src/run_single_simulation.py' n_scenarios=$n_scenarios distance_metric=$method wl_constraint_type=$wl_constraint_type var_wl=$var_wl cvar_wl=$cvar_wl cvar_alpha=$cvar_alpha p_chance=$p_chance tree_based=$tree_based max_threads=$max_threads start_from_scratch=$start_from_scratch")


if __name__=='__main__':
    i=1
    for n in n_scenarios:
        for var in vars:
            for alpha in alphas:
                for cvar in cvars:
                    for tree in tree_based:
                        make_slurm_job(
                            name=f'optrun_{i}.sh', 
                            n=n, 
                            var=var, 
                            cvar=cvar, 
                            alpha=alpha, 
                            tree=tree)
                        i+=1