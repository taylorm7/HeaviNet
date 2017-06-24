#!/bin/bash

LEVELS=7

ACTION=$1
SONG=$2
dot="$(cd "$(dirname "$0")"; pwd)"

SONGPATH="$dot/data/songs/$SONG"
DATAPATH="$dot/data/$SONG.data"
MATLABCODE="$dot/matlab_code"
FINISHPATH="$DATAPATH/SONG.wav"


#usage

# "format" $SONG $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
# "load" $SONG $RECEPTIVE_FIELD
# "train" $SONG $RECEPTIVE_FIELD $EPOCHS
# "generate" $SONG $SEED $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
# "run/train_gen" $SONG $SEED $RECEPTIVE_FIELD $EPOCHS $DOWNSAMPLE_RATE

if [ $ACTION = "generate" ] || [ $ACTION = "train_generate" ] || [ $ACTION = "run" ]; then
	SEED=$3
	SEEDPATH="$dot/data/songs/$SEED"
	FINISHPATH="$DATAPATH/SONG.wav"
	if [ -z $4 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$4
	fi
	if [ $ACTION = "generate" ]; then
		if [ -z $5 ]; then
			DOWNSAMPLE_RATE=0
		else
			DOWNSAMPLE_RATE=$5
		fi
	else
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
	fi
else
	if [ -z $3 ]; then
		RECEPTIVE_FIELD=1
	else
		RECEPTIVE_FIELD=$3
	fi

	if [ $ACTION = "train" ]; then
		if [ -z $4 ]; then
			EPOCHS=1
		else
			EPOCHS=$4
		fi
	else
		if [ -z $4 ]; then
			DOWNSAMPLE_RATE=0
		else
			DOWNSAMPLE_RATE=$4
		fi
		if [ -z $5 ]; then
			EPOCHS=1
		else
			EPOCHS=$5
		fi
	fi

fi

echo "Call: $ACTION [Seed:$SEED] [Receptive field:$RECEPTIVE_FIELD] [Epochs:$EPOCHS] [Downsample rate:$DOWNSAMPLE_RATE]"

MATLABSONG="$DATAPATH/matlab_song_r$RECEPTIVE_FIELD.mat"


if [ $ACTION = "format" ]; then
	echo "Formatting song:$SONG at:$SONGPATH with levels:$LEVELS and downsample rate:$DOWNSAMPLE_RATE"
	if [ -f $SONGPATH ]; then
		mkdir $DATAPATH
		echo "Data directory at '$DATAPATH'"
		~/Matlab/matlab -nojvm -sd "$MATLABCODE" -r "audio_format('$SONGPATH', '$DATAPATH', $DOWNSAMPLE_RATE, $LEVELS, $RECEPTIVE_FIELD ); quit;"
		#~/Matlab/matlab -nojvm -r "audio_formatting(7, '$SONGPATH', '$MATLABSONG'  ); quit;"
		echo "Matlab formatting stored at $MATLABSONG"
	else
		echo "The file '$SONG' not found at '$SONGPATH'"
		echo "Make sure song_name.wav is located in ./data/songs/"
	fi
elif [ $ACTION = "load" ]; then
	if [ -f $MATLABSONG ]; then
		echo "Loading song:$SONG in $MATLABSONG"
		python heavinet.py $ACTION $DATAPATH $RECEPTIVE_FIELD
	else
		echo "The file '$SONG' not found at '$MATLABSONG'"
		echo "Try loading with ./run_heavinet.sh load song_name.mp3"
	fi	
elif [ $ACTION = "train" ]; then
	if [ -f $MATLABSONG ]; then
		echo "Training on song $SONG in $MATLABSONG"
		for (( i=0; i<$LEVELS; i++ ))	
		do
			python heavinet.py $ACTION $DATAPATH $i $RECEPTIVE_FIELD $EPOCHS
		done
	else
		echo "The file '$SONG' not found at '$MATLABSONG'"
		echo "Try loading with ./run_heavinet.sh load song_name.mp3"
	fi
elif [ $ACTION = "generate" ]; then
	if [[ -f $MATLABSONG && -f $SEEDPATH ]]; then
		echo "Generating on song $SONG from seed $SEED"
		echo "Data path:$DATAPATH"

		GENSEEDNAME="seed_0_r$RECEPTIVE_FIELD"
		GENSEEDPATH="$DATAPATH/$GENSEEDNAME.mat"


		~/Matlab/matlab -nojvm -sd "$MATLABCODE" -r "audio_seed(0, '$SEEDPATH', '$GENSEEDPATH', $DOWNSAMPLE_RATE, $LEVELS, $RECEPTIVE_FIELD ); quit;"
		
		for ((I=1 ; I<=LEVELS ; I++)); do


			GENSONGNAME="song_$I""_r$RECEPTIVE_FIELD"
			GENSONGPATH="$DATAPATH/$GENSONGNAME.mat"

			python heavinet.py $ACTION $DATAPATH $GENSEEDPATH $(($I-1)) $RECEPTIVE_FIELD
		
			GENSEEDNAME="seed_$I""_r$RECEPTIVE_FIELD"
			GENSEEDPATH="$DATAPATH/$GENSEEDNAME.mat"
			echo "$I $LEVELS"
			if [ $I != $LEVELS ]; then
				echo "filter level"
				~/Matlab/matlab -nojvm -sd "$MATLABCODE" -r "filter_level('$GENSONGPATH', '$GENSEEDPATH', $I, $RECEPTIVE_FIELD ); quit;"
			else
				echo "finish song"
				~/Matlab/matlab -nojvm -sd "$MATLABCODE" -r "audio_finish('$GENSONGPATH', '$FINISHPATH', '$SONGPATH', $LEVELS, $DOWNSAMPLE_RATE ); quit;"
			fi
		done
		
		#~/Matlab/matlab -nojvm -sd "$MATLABCODE" -r "audio_finish('$GENSEEDPATH', '$FINISHPATH', '$SONGPATH', $LEVELS, $DOWNSAMPLE_RATE ); quit;"

	else
		echo "The file '$SONGPATH' or '$SEEDPATH' is not valid"
		echo "First try loading with ./run_heavinet.sh load song_name.mp3"
		echo "Then training with ./run_heavinet.sh train song_name.mp3"
	fi

elif [ $ACTION = "run" ]; then
	if [[ -f $SONGPATH && -f $SEEDPATH ]]; then
		./$0 "format" $SONG $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
		./$0 "load" $SONG $RECEPTIVE_FIELD
		./$0 "train" $SONG $RECEPTIVE_FIELD $EPOCHS
		./$0 "generate" $SONG $SEED $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
		echo "Compled Run"

	else
		echo "Invalid $SONG or $SEED"
	fi

elif [ $ACTION = "train_generate" ]; then
	if [[ -f $MATLABSONG && -f $SEEDPATH ]]; then
		./$0 "train" $SONG $RECEPTIVE_FIELD $EPOCHS
		./$0 "generate" $SONG $SEED $RECEPTIVE_FIELD $DOWNSAMPLE_RATE
		echo "Compled Run"

	else
		echo "Invalid $SONG or $SEED"
	fi

else
	echo "Please enter an action, 'load song.mp3', 'train song.wav', or 'generate song.mp4 seed.mp3'"
fi

#rm -d DATAPATH

#~/Matlab/matlab -nojvm -r 'upsample_level("a", 1); quit;'
#~/Matlab/matlab -nojvm -r 'try upsample_level('a', 1); catch; end; quit'

