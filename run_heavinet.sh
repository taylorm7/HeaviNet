#!/bin/bash

ACTION=$1

SONG=$2

SONGPATH="./data/songs/$SONG"
DATAPATH="./data/$SONG.data"

MATLABSONG="$DATAPATH/matlab_song.mat"
MATLABSEED="$DATAPATH/matlab_seed.mat"

if [ $ACTION = "load" ]; then
	echo "Loading song:$SONG at:$SONGPATH"
	if [ -f $SONGPATH ]; then
		echo "Formatting '$SONG':"
		mkdir $DATAPATH
		echo "Data directory at '$DATAPATH'"
		~/Matlab/matlab -nojvm -r "audio_formatting(7, '$SONGPATH', '$MATLABSONG'  ); quit;"
		echo "Matlab formatting stored at $MATLABSONG"
	else
		echo "The file '$SONG' not found at '$SONGPATH'"
		echo "Make sure song_name.wav is located in ./data/songs/"
	fi
elif [ $ACTION = "train" ]; then
	RECEPTIVE_FIELD=$3
	if [ -f $MATLABSONG ]; then
		echo "Training on song $SONG in $MATLABSONG"
		python heavinet.py $ACTION $DATAPATH $RECEPTIVE_FIELD
	else
		echo "The file '$SONG' not found at '$MATLABSONG'"
		echo "Try loading with ./run_heavinet.sh load song_name.mp3"
	fi
elif [ $ACTION = "generate" ]; then
	echo "Generating..."
	SEED=$3
	SEEDPATH="./data/songs/$SEED"

else
	echo "Please enter an action, 'load song.mp3', 'train song.wav', or 'generate song.mp4 seed.mp3'"
fi

#rm -d DATAPATH

#~/Matlab/matlab -nojvm -r 'upsample_level("a", 1); quit;'
#~/Matlab/matlab -nojvm -r 'try upsample_level('a', 1); catch; end; quit'

