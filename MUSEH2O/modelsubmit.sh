#!/bin/bash
#SBATCH -D /storage/work/your/path		      # your working directory
#SBATCH -o /storage/work/your/path/to /Susquehanna_Notebooks/output # Name of the output
#SBATCH --job-name=SusquehannaModel         # give the job a name
##SBATCH --account=azh5924_b		        	  # Only use once your job works
##SBATCH --partition=sla-prio               # Only use once your job works
#SBATCH --account=open                      # FOR TESTING – specify account
#SBATCH --partition=open                    # FOR TESTING – specify partition
#SBATCH --nodes=1                           # request a node
#SBATCH --ntasks=1                          # request a task / cpu
#SBATCH --cpus-per-task=48                  # cpu per task
#SBATCH --mem-per-cpu=1G                    # request the memory per node
#SBATCH --time=110:00:00                    # set a run time limit
##SBATCH --time=03:00:00                    # FOR TESTING- set a timelimit
#SBATCH --mail-user=your email              # address for email notification
#SBATCH --mail-type=ALL                     # email at Begin and End of job

# load modules here

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# execution code here
srun python ./main_susquehanna.py