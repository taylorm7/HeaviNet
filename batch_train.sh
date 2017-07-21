#!/bin/bash

# At most XX minute of time
#PBS -l walltime=40:00:00    

# One core on any number of nodes
#PBS -l procs=1,gpus=2

#newriver cluster
#PBS -W group_list=newriver  

# queue
#PBS -q p100_normal_q

#PBS -A heavinet

# write output and error to the same file
#PBS -j oe

# Email when your job starts, completes, or aborts
#PBS -M taylorm7@vt.edu
#PBS -m ea

module purge

module load gcc/5.2.0 cuda/8.0.61

export PATH=$PATH:/home/taylorm7/opt_py35/bin 
export PYTHONUSERBASE=/home/taylorm7/opt_py35/newriver/python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/cuda/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/opt/lib/

dot=$PBS_O_WORKDIR
cd $dot
pwd

export dot=$dot
./run_heavinet.sh train beethoven_7.wav 7 100

echo "Code finished!"

