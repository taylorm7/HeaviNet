#!/bin/bash

rm $1

level_start=$2
level_stop=$3
num_levels=$4


echo '#!/bin/bash' >> $1
echo '#PBS -l walltime=10:00:00' >> $1
echo '#PBS -l procs=1,gpus=2' >> $1
echo '#PBS -W group_list=newriver' >> $1
echo '#PBS -q p100_normal_q' >> $1
echo '#PBS -A heavinet' >> $1
echo '#PBS -j oe' >> $1
if [ "$level_stop" == "$num_levels" ]; then
	echo '#PBS -M taylorm7@vt.edu' >> $1
	echo '#PBS -m ea' >> $1
fi

echo 'module purge' >> $1

echo 'module load gcc/5.2.0 cuda/8.0.61' >> $1

echo 'export PATH=$PATH:/home/taylorm7/opt_py35/bin' >> $1
echo 'export PYTHONUSERBASE=/home/taylorm7/opt_py35/newriver/python' >> $1

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/cuda/lib64/' >> $1
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taylorm7/opt/lib/' >> $1


echo 'dot=$PBS_O_WORKDIR' >> $1
#echo 'dot=/home/sable/HeaviNet' >> $1

echo 'cd $dot' >> $1
echo 'pwd' >> $1

echo 'export dot=$dot' >> $1

epochs=$5
receptive_field=$6
song=$7

for (( i=$level_start; i<$level_stop; i++ ))
do
	echo "./run_heavinet.sh train $song $receptive_field $epochs $i" >> $1
done

if [ "$level_stop" == "$num_levels" ]; then
	echo "sleep 10m" >> $1
	echo "./run_heavinet.sh generate beethoven_7.wav beethoven_7.wav 14" >> $1
	echo "./run_heavinet.sh generate beethoven_7.wav bach_10.wav 14" >> $1
fi


