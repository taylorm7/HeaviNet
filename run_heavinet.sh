#!/bin/bash

LEVELS=8

#usage
# "format" $SONG $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
# "load" $SONG $RECEPTIVE_FIELD
# "train" $SONG $RECEPTIVE_FIELD $EPOCHS $TRAIN_START
# "generate" $SONG $SEED $RECEPTIVE_FIELD $DOWNSAMPLE_RATE $GENERATE_START
# "run/train_gen" $SONG $SEED $RECEPTIVE_FIELD $EPOCHS $DOWNSAMPLE_RATE $GENERATE_START

if [ -z $dot ]; then
	# regular call with matlab script at ~/Matlab/matlab
	dot="$(cd "$(dirname "$0")"; pwd)"
	echo "Regular call:$dot"
	MATLABCALL=~/Matlab/matlab
else
	# ARC batch call
	echo "Batch call:$dot"
	MATLABCALL=matlab
fi

cd $dot

ACTION=$1
SONG=$2

SONGPATH="$dot/data/songs/$SONG"
DATAPATH="$dot/data/$SONG.data"

if [ $ACTION = "format" ]; then
	if [ -z $3 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$3
	fi
	if [ -z $4 ]; then
		DOWNSAMPLE_RATE=0
	else
		DOWNSAMPLE_RATE=$4
	fi
elif [ $ACTION = "load" ]; then
	if [ -z $3 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$3
	fi
elif [ $ACTION = "train" ]; then
	if [ -z $3 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$3
	fi
	if [ -z $4 ]; then
		EPOCHS=1
	else
		EPOCHS=$4
	fi
	if [ -z $5 ]; then
		TRAIN_START=0
		train_all_levels=1
	else
		TRAIN_START=$5
		train_all_levels=0
	fi

elif [ $ACTION = "generate" ]; then
	SEED=$3
	SEEDPATH="$dot/data/songs/$SEED"
	if [ -z $4 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$4
	fi
	if [ -z $5 ]; then
		DOWNSAMPLE_RATE=0
	else
		DOWNSAMPLE_RATE=$5
	fi
	if [ -z $6 ]; then
		GENERATE_START=0
	else
		GENERATE_START=$6
	fi
elif [ $ACTION = "train_generate" ] || [ $ACTION = "run" ]; then
	SEED=$3
	SEEDPATH="$dot/data/songs/$SEED"
	if [ -z $4 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$4
	fi
	if [ -z $5 ]; then
		EPOCHS=1
	else
		EPOCHS=$5
	fi
	if [ -z $6 ]; then
		DOWNSAMPLE_RATE=0
	else
		DOWNSAMPLE_RATE=$6
	fi
	if [ -z $7 ]; then
		GENERATE_START=0
	else
		GENERATE_START=$7
	fi
fi

echo "Call: $ACTION [Seed:$SEED] [Receptive field:$RECEPTIVE_FIELD] [Epochs:$EPOCHS] [Downsample rate:$DOWNSAMPLE_RATE]"
echo "[Train start:$TRAIN_START] [Train all levels:$train_all_levels] [Generate start:$GENERATE_START]"

MATLABSONG="$DATAPATH/matlab_song_r$RECEPTIVE_FIELD.mat"

if [ $ACTION = "format" ]; then
	SECONDS=0
	echo "Formatting song:$SONG at:$SONGPATH with levels:$LEVELS and downsample rate:$DOWNSAMPLE_RATE"
	if [ -f $SONGPATH ]; then
		mkdir $DATAPATH
		echo "Data directory at '$DATAPATH'"
		$MATLABCALL -nojvm -r "try, audio_format('$SONGPATH', '$DATAPATH', $DOWNSAMPLE_RATE, $LEVELS, $RECEPTIVE_FIELD ); , catch ME, error_msg = getReport(ME); disp(error_msg), end, exit"
		echo "Matlab formatting stored at $MATLABSONG"
	else
		echo "The file '$SONG' not found at '$SONGPATH'"
		echo "Make sure song_name.wav is located in ./data/songs/"
	fi
	duration=$SECONDS
	echo "Format: $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
elif [ $ACTION = "load" ]; then
	SECONDS=0
	if [ -f $MATLABSONG ]; then
		echo "Loading song:$SONG in $MATLABSONG"
		python3 heavinet.py $ACTION $DATAPATH $RECEPTIVE_FIELD
	else
		echo "The file '$SONG' not found at '$MATLABSONG'"
		echo "Try loading with ./run_heavinet.sh load song_name.mp3"
	fi
	duration=$SECONDS
	echo "Load: $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
elif [ $ACTION = "train" ]; then
	SECONDS=0
	if [ -f $MATLABSONG ]; then

		echo "Training on song $SONG in $MATLABSONG"
		for (( i=$TRAIN_START; i<$LEVELS; i++ ))	
		do
			echo " running level $i in background process..."
			level_seconds=$SECONDS
			python3 heavinet.py $ACTION $DATAPATH $i $RECEPTIVE_FIELD $EPOCHS $LEVELS >> "$DATAPATH/$i.txt" 2>&1 #& #parallel
			level_duration=$(($SECONDS-level_seconds))
			echo "Level duration: $(($level_duration / 60)) minutes and $(($level_duration % 60)) seconds elapsed."
			if [ $train_all_levels == 0 ]; then
				break
			fi
		done
		wait
		echo "Training finished"
		for (( i=$TRAIN_START; i<$LEVELS; i++ ))	
		do
			cat "$DATAPATH/$i.txt"
			echo ""
			if [ $train_all_levels == 0 ]; then
				break
			fi
		done
	else
		echo "The file '$SONG' not found at '$MATLABSONG'"
		echo "Try loading with ./run_heavinet.sh load song_name.mp3"
	fi
	duration=$SECONDS
	echo "Train: $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
elif [ $ACTION = "generate" ]; then
	SECONDS=0
	if [[ -f $MATLABSONG && -f $SEEDPATH ]]; then
		echo "Generating on song $SONG from seed $SEED"
		echo "Data path:$DATAPATH"

		GENSEEDNAME="seed_$GENERATE_START""_r$RECEPTIVE_FIELD"
		GENSEEDPATH="$DATAPATH/$GENSEEDNAME.mat"

		$MATLABCALL -nojvm -r "try, audio_seed($GENERATE_START, '$SEEDPATH', '$GENSEEDPATH', $DOWNSAMPLE_RATE, $LEVELS, $RECEPTIVE_FIELD, '$DATAPATH' ); , catch ME, error_msg = getReport(ME); disp(error_msg), end, exit"
		GENERATE_START=$((GENERATE_START+1))
		for ((I=$GENERATE_START ; I<=LEVELS ; I++)); do
			GENSONGNAME="song_$I""_r$RECEPTIVE_FIELD"
			GENSONGPATH="$DATAPATH/$GENSONGNAME.mat"
			GENSONGFILE="$DATAPATH/$GENSONGNAME.wav"

			python3 heavinet.py $ACTION $DATAPATH $GENSEEDPATH $(($I-1)) $RECEPTIVE_FIELD $LEVELS
			
			GENSEEDNAME="seed_$I""_r$RECEPTIVE_FIELD"
			GENSEEDPATH="$DATAPATH/$GENSEEDNAME.mat"
			
			$MATLABCALL -nojvm -r "try, filter_level('$GENSONGPATH', '$GENSEEDPATH', $I, $RECEPTIVE_FIELD, '$DATAPATH' ); , catch ME, error_msg = getReport(ME); disp(error_msg), end, exit"
			
		done
		
	else
		echo "The file '$SONGPATH' or '$SEEDPATH' is not valid"
		echo "First try loading with ./run_heavinet.sh load song_name.mp3"
		echo "Then training with ./run_heavinet.sh train song_name.mp3"
	fi
	duration=$SECONDS
	echo "Generate: $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
elif [ $ACTION = "run" ]; then
	SECONDS=0
	if [[ -f $SONGPATH && -f $SEEDPATH ]]; then
		./$0 "format" $SONG $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
		./$0 "load" $SONG $RECEPTIVE_FIELD
		./$0 "train" $SONG $RECEPTIVE_FIELD $EPOCHS
		./$0 "generate" $SONG $SEED $RECEPTIVE_FIELD $DOWNSAMPLE_RATE $GENERATE_START
		echo "Compled Run"

	else
		echo "Invalid $SONG or $SEED"
	fi
	duration=$SECONDS
	echo "Run: $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
elif [ $ACTION = "train_generate" ]; then
	SECONDS=0
	if [[ -f $MATLABSONG && -f $SEEDPATH ]]; then
		./$0 "train" $SONG $RECEPTIVE_FIELD $EPOCHS
		./$0 "generate" $SONG $SEED $RECEPTIVE_FIELD $DOWNSAMPLE_RATE $GENERATE_START
		echo "Compled Run"

	else
		echo "Invalid $SONG or $SEED"
	fi
	duration=$SECONDS
	echo "Train/Gen: $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
else
	echo "Please enter an action, 'load song.mp3', 'train song.wav', or 'generate song.mp4 seed.mp3'"
fi

