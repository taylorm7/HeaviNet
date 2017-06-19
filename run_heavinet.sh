#!/bin/bash

SONG=$1
SONGPATH="./data/songs/$SONG"
DATAPATH="./data/$SONG.data"

MATLABSONG="$DATAPATH/matlab_song.mat"
MATLABSEED="$DATAPATH/matlab_seed.mad"


echo "Song:$SONG at:$SONGPATH"

if [ -f $SONGPATH ]; then
	echo "Formatting '$SONG':"
else
	echo "The file '$SONG' not found at '$SONGPATH'"
fi

#rm -d DATAPATH
mkdir $DATAPATH

echo "Data directory at '$DATAPATH'"

~/Matlab/matlab -nojvm -r "audio_formatting(7, '$SONGPATH', '$MATLABSONG'  ); quit;"

python heavinet.py

#~/Matlab/matlab -nojvm -r 'upsample_level("a", 1); quit;'
#~/Matlab/matlab -nojvm -r 'try upsample_level('a', 1); catch; end; quit'

echo "Matlab Audio Formatting Completed"

