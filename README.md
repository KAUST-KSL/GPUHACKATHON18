# GPUHACKATHON18

Programming GPU with OpenACC
The introduction talk from Dr Saber Feki can be downloaded 

1. Connection to IBEX:

```
ssh username@glogin.ibex.kaust.edu.sa
```

Go to the following location:

```
cd /scratch/dragon/amd/$USER 
mkdir gpuhackathon18 
cd gpuhackathon18
```

From now on we consider that you are always inside the folder /scratch/dragon/amd/$USER/gpuhackathon18

Before you start working on IBEX, load the following modules:

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

Compilation

Example:
pgf90  -O2 -ta=tesla -acc -Minfo=accel ...

Options: -ta=tesla:lineinfo -Minfo=all,intensity

lineinfo
It will provide the lines in the code that you have memory problems
Minfo=all
It will provide all the compiler information (including the acceleration), for example
Loop not vectorized: loop count too small
Loop unrolled 6 times (completely unrolled)
Minfo=intensity
Provides the intensity of all the loops, intensity is the (Compute operations/Memory Operations), if it is more or equal to 1.0 then we should move this loop to GPUs otherwise not.
For example,
   210, Possible copy in and copy out of q in call to coef_df4_premiere
         Intensity = 0.50
    442, Intensity = 1.67
Profiling
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



