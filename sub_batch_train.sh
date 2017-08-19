#!/bin/bash

num_levels=8
batches=8

epochs=200
receptive_field=16
song=beethoven_7.wav
generate=choir_15.wav

train_time=40:00:00

step_size=$(($num_levels / $batches))

for (( i=0; i<$num_levels; i+=$step_size ))
do
name="job_$song""_$i.sh"
./sub_create.sh $name $i $(($i+$step_size)) $num_levels $epochs $receptive_field $song $generate $train_time
chmod +x $name
qsub ./$name
done

