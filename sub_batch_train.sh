#!/bin/bash

num_levels=8
batches=4

step_size=$(($num_levels / $batches))

for (( i=0; i<$num_levels; i+=$step_size ))
do
name="job_$i.sh"
./sub_create.sh $name $i $(($i+$step_size)) $num_levels
chmod +x $name
./$name
done

