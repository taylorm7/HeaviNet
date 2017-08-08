#!/bin/bash

num_levels=8
batches=8

epochs=50
receptive_field=14
song=beethoven_7.wav

step_size=$(($num_levels / $batches))

for (( i=0; i<$num_levels; i+=$step_size ))
do
name="job_$i.sh"
./sub_create.sh $name $i $(($i+$step_size)) $num_levels $epochs $receptive_field $song
chmod +x $name
qsub ./$name
done

