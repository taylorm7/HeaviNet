#!/bin/bash

# At most XX minute of time
#PBS -l walltime=4:00:00    

# One core on any number of nodes
#PBS -l procs=1,gpus=1

#newriver cluster
#PBS -W group_list=newriver  

# queue
#PBS -q vis_q

#PBS -A heavinet

# write output and error to the same file
#PBS -j oe

# Email when your job starts, completes, or aborts
#PBS -M taylorm7@vt.edu
#PBS -m ea

module purge

module load gcc/5.2.0 cuda/8.0.61 matlab/R2016b

export PATH=$PATH:/home/taylorm7/opt_py35/bin
# gpu
export PYTHONUSERBASE=/home/taylorm7/opt_py35/newriver/python
# cpu
#export PYTHONUSERBASE=/home/taylorm7/opt_py35/cpu_tf

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/cuda/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/opt/lib/
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/taylorm7/opt/lib/pkgconfig

dot=$PBS_O_WORKDIR
cd $dot
pwd

export dot=$dot

iterations=5

song=beethoven_7.wav
target=rand.wav
receptive_field=16

seed=$target
name=$(echo $target | cut -f 1 -d '.')
extension=$(echo $target | cut -f 2 -d '.')

for ((i=0 ; i<iterations ; i++)); do
	./run_heavinet.sh load $song $seed $receptive_field
	./run_heavinet.sh generate $song $seed $receptive_field
	./run_heavinet.sh create $song $seed $receptive_field

	nextSeed="$name""_$(($i+1)).$extension"
	cp ./data/$song.data/$seed.seed/song_r$receptive_field.wav ./data/songs/$nextSeed
	seed=$nextSeed
done
echo "Code finished!"

