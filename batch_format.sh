#!/bin/bash

# At most XX minute of time
#PBS -l walltime=02:15:00    

# One core on any number of nodes
#PBS -l procs=1

#newriver cluster
#PBS -W group_list=newriver  

# queue
#PBS -q normal_q

#PBS -A heavinet

# write output and error to the same file
#PBS -j oe

# Email when your job starts, completes, or aborts
#PBS -M taylorm7@vt.edu
#PBS -m ea

module purge

module load gcc/5.2.0 cuda/8.0.61 matlab/R2016b

export PATH=$PATH:/home/taylorm7/opt_py35/bin 
export PYTHONUSERBASE=/home/taylorm7/opt_py35/cpu_tf

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/taylorm7/opt/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/opt/lib/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/cuda/lib64/

dot=$PBS_O_WORKDIR
cd $dot
pwd

export dot=$dot
./run_heavinet.sh format choir_15.wav 16
./run_heavinet.sh load choir_15.wav choir_15.wav 16

./run_heavinet.sh format choir_15.wav 32
./run_heavinet.sh load choir_15.wav choir_15.wav 64

./run_heavinet.sh format choir_15.wav 64
./run_heavinet.sh load choir_15.wav choir_15.wav 64

./run_heavinet.sh format choir_15.wav 128
./run_heavinet.sh load choir_15.wav choir_15.wav 128

./run_heavinet.sh format choir_15.wav 256
./run_heavinet.sh load choir_15.wav choir_15.wav 256

./run_heavinet.sh format bach.wav 64
./run_heavinet.sh load bach.wav beethoven_7.wav 64


#./run_heavinet.sh load choir_15.wav choir_15.wav 16
#./run_heavinet.sh load choir_15.wav choir_15.wav 16
#./run_heavinet.sh load choir_15.wav rand.wav 16

#./run_heavinet.sh load choir_15.wav beethoven_a.wav 100

echo "Code finished!"

