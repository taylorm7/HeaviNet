#!/bin/bash

num_levels=7
batches=7

epochs=10000
receptive_field=1
song=beethoven_7.wav
generate=bach_10.wav

train_time=40:00:00

step_size=$(($num_levels / $batches))

for (( i=0; i<$num_levels; i+=$step_size ))
do
name="job_$song""_$i.sh"
./sub_create.sh $name $i $(($i+$step_size)) $num_levels $epochs $receptive_field $song $generate $train_time
chmod +x $name
qsub ./$name
done

