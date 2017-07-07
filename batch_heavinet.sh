#!/bin/bash

# At most XX minute of time
#PBS -l walltime=00:30:00    

# One core on any number of nodes
#PBS -l procs=7,gpus=1

#newriver cluster
#PBS -W group_list=newriver  

# open queue
#PBS -q open_q

# write output and error to the same file
#PBS -j oe

# Email when your job starts, completes, or aborts
#PBS -M taylorm7@vt.edu
#PBS -m bea

module purge

module load gcc/5.2.0 python/3.5.0 cuda/8.0.44 atlas/3.11.36 matlab/R2016b

export PYTHONUSERBASE=/home/taylorm7/newriver/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/cuda/lib64/

dot=$PBS_O_WORKDIR
cd $dot
pwd

./run_heavinet.sh run bach_10.mp3 bach_10.mp3

echo "Code finished!"

