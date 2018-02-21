# KAUST GPU HACKATHON 2018

Programming GPU with OpenACC

The introduction talk from Dr Saber Feki can be downloaded [here](material/Hackathon2018_OpenACC.pdf)

1. Connection to IBEX:

```
ssh -X username@glogin.ibex.kaust.edu.sa
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

4. Compilation

Example:

Serial Laplace:

In case that you plan to use PGI compiler:
```
pgcc -ta=tesla:cc60 -O2 -Minfo=all -o laplace_serial src/laplace_serial.c
```
* -Minfo=all: Informs you about all the messages 

Output:
```
main:
     29, Loop not vectorized/parallelized: contains call
     32, Generated an alternate version of the loop
         Generated vector simd code for the loop
         Generated 3 prefetch instructions for the loop
         Generated vector simd code for the loop
         Generated 3 prefetch instructions for the loop
     41, Generated vector simd code for the loop containing reductions
         Generated 2 prefetch instructions for the loop
initialize:
     68, Memory zero idiom, loop replaced by call to __c_mzero8
     73, Generated vector simd code for the loop
         Residual loop unrolled 2 times (completely unrolled)
     78, Generated vector simd code for the loop
         Residual loop unrolled 2 times (completely unrolled)
track_progress:
     90, Loop not vectorized/parallelized: contains call
```

5. Execution

Submission script:
```
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
srun -n 1 --hint=nomultithread ./laplace_serial
```

Submit:
```
sbatch submit_laplace_serial.sh
```

Source: [submission_laplace_serial](submit_laplace_serial.sh)

6. Profiling

Compile your code for CPU (remove -acc and -ta from the PGI compilation if they were included)

Execute:
```
sbatch submit_profiling_terminal.sh
```
Source: [submission_profile_terminal](submit_profile_terminal.sh)

Open the output file and see the profiling information.

To use a GUI:
* Use the submission file [submission_profile_file](submit_profile_file.sh)

* Execute:
```
nvvp results.nvprof
```

7. Laplace version with initial OpenACC pragmas

```
pgcc -O2 -ta=tesla:cc60 -acc -Minfo=accel -o laplace_bad_acc src/laplace_bad_acc.c
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
You can modify the time limit and the number of the tasks, if required.

Use compiler pgf90 for Fotran and pgc++ for C++

Options: -ta=tesla:cc60 -Minfo=all,intensity


* Minfo=all
It will provide all the compiler information (including the acceleration), for example
Loop not vectorized: loop count too small
Loop unrolled 6 times (completely unrolled)

* Minfo=intensity
Provides the intensity of all the loops, intensity is the (Compute operations/Memory Operations), if it is more or equal to 1.0 then we should move this loop to GPUs, otherwise not.


8. Execution

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
srun -n 1 --hint=nomultithread ./laplace_bad_acc
```

Submit:
```
sbatch submit_laplace_bad_acc.sh
```

Source: [submission_laplace_bad_acc](submit_laplace_bad_acc.sh)


Modify the X according to your team number (1-6)
In the above exampe we want to use one Nvidia P100 card, if you plan to use 2 cards then declare:

```
#SBATCH --gres=gpu:p100:2
```

9. Profiling

Adjust the name of the binary in all job scripts

Execute:
```
sbatch submit_profiling_terminal.sh
```
Source: [submission_profile_terminal](submit_profile_terminal.sh)

Open the output file and see the profiling information.

To use a GUI:
* Use the submission file [submission_profile_file](submit_profile_file.sh)

* Execute:
```
nvvp results.nvprof
```

10. Latest version with optimized OpenACC pragmas

Repeat the previous instructions with file [laplace_final_acc.c](src/laplace_final_acc.c)

# Material

[Ibex cheat sheet](material/ibex_flyer.pdf)
[OpenACC web page](https://www.openacc.org)

[OpenACC reference guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC%20API%202.6%20Reference%20Guide.pdf)

[OpenACC programming guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0.pdf)

[OpenACC getting started guide](http://www.pgroup.com/doc/openacc17_gs.pdf)

[PGI	Compiler guide](http://www.pgroup.com/resources/docs/18.1/pdf/pgi18ug-x86.pdf)

[CUDA	Cuda with C/C++](https://developer.nvidia.com/how-to-cuda-c-cpp)
[Cuda with Fortran](https://developer.nvidia.com/cuda-fortran)

[GPU Libraries	GPUs libraries](https://developer.nvidia.com/how-to-cuda-libraries)

[Matlab and GPU](https://developer.nvidia.com/matlab-cuda)


Deep Neural Networks - Cudnn
Load:
```
module load cudnn
```



