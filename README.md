# KAUST GPU HACKATHON 2018

Programming GPU with OpenACC

The introduction talk from Dr Saber Feki can be downloaded here

1. Connection to IBEX:

```
ssh username@glogin.ibex.kaust.edu.sa
```

2. Go to the following location:

```
cd /scratch/dragon/amd/$USER 
mkdir gpuhackathon18 
cd gpuhackathon18
```

From now on we consider that you are always inside the folder /scratch/dragon/amd/$USER/gpuhackathon18

3. Before you start working on IBEX, load the following modules:

```
module load cuda/9.0.176
module load pgi/17.10 
```

or by hand execute the following
export PATH=$PATH:/usr/local/cuda/bin/
export PATH=/sw/cs/pgi/linux86-64/16.1/mpi/openmpi-1.10.1/bin/:$PATH
export PATH=$PATH:/sw/cs/pgi/linux86-64/16.1/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/cs/pgi/linux86-64/16.1/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/cs/pgi/linux86-64/16.1/mpi/openmpi-1.10.1/lib/
export PATH=$PATH:/opt/allinea/forge/bin/

4. Compilation

Example:

In case that you plan to use PGI compiler:
```
pgcc -O2 -ta=tesla:cuda8.0 -acc -Minfo=accel -o test test1.c
```
Flags:
* -acc: activates the OpenACC compilation
* -Minfo=accel: Informs you about the accelerators messages 

Output:
```
main:
     31, Generating implicit copyout(Temperature[1:1000][1:1000])
         Generating implicit copyin(Temperature_previous[:][:])
     32, Loop is parallelizable
     33, Loop is parallelizable
         Accelerator kernel generated
         Generating Tesla code
         32, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
         33, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
     41, Generating implicit copy(Temperature_previous[1:1000][1:1000])
         Generating implicit copyin(Temperature[1:1000][1:1000])
     42, Loop is parallelizable
     43, Loop is parallelizable
         Accelerator kernel generated
         Generating Tesla code
         42, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
         43, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
         44, Generating implicit reduction(max:worst_dt)
```

Use compiler pgf90 for Fotran and pgc++ for C++

Options: -ta=tesla:cuda8.0:lineinfo -Minfo=all,intensity

* lineinfo:
It will provide the lines in the code that you have memory problems

* Minfo=all
It will provide all the compiler information (including the acceleration), for example
Loop not vectorized: loop count too small
Loop unrolled 6 times (completely unrolled)

* Minfo=intensity
Provides the intensity of all the loops, intensity is the (Compute operations/Memory Operations), if it is more or equal to 1.0 then we should move this loop to GPUs, otherwise not.

5. Execution

Submission script:
```
#!/bin/bash 
#SBATCH --partition=batch 
#SBATCH --job-name="test" 
#SBATCH --gres=gpu:p100:1
#SBATCH --res=HACKATHON_TEAMX
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --time=00:10:00 
#SBATCH --exclusive 
#SBATCH --err=JOB.%j.err 
#SBATCH --output=JOB.%j.out 
#--------------------------------------------# 
module load cuda/9.0.176
module load pgi/17.10
srun -n 1 --hint=nomultithread ./test
```

Modify the X according to your team number (1-6)
In the above exampe we want to use one Nvidia P100 card, if you plan to use 2 cards then declare:

```
#SBATCH --gres=gpu:p100:2
```

You can modify the time limit and the number of the tasks, if required.

Source: [submission_file](submit.sh)

6. Profiling
You compile your code for CPU

Execute:
nvprof --cpu-profiling on ./executable
Profiling instructions
Use the tool nvvp for GUI
Profiling with Allinea
Allinea profiling tool: /opt/allinea/forge/

Execution
export CUDA_VISIBLE_DEVICES=X
check which GPUS are used:
nvidia-smi
Material
OpenACC	OpenACC web page
OpenACC reference guide
OpenACC programming guide
OpenACC getting started guide

PGI	Compiler guide

CUDA	Cuda with C/C++
Cuda with Fortran

GPU Libraries	GPUs libraries

Other	Matlab and GPU


AMGX and MiniFE
AMGX and MiniFE installation instructions (if needed only): http://hpc.kaust.edu.sa/AMGX_MINIFE_GPU
Deep Neural Networks - Cudnn
Available here: module load cudnn



