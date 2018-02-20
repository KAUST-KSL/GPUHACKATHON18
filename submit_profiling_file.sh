#!/bin/bash 
#SBATCH --partition=batch 
#SBATCH --job-name="test" 
#SBATCH --gres=gpu:p100:1
#SBATCH --res=HACKATHON_TEAMS
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --time=00:10:00 
#SBATCH --exclusive 
#SBATCH --err=JOB.%j.err 
#SBATCH --output=JOB.%j.out 
#--------------------------------------------# 
module load cuda/9.0.176
module load pgi/17.10
srun -n 1 nvprof -o results.nvprof --cpu-profiling on ./laplace_serial
