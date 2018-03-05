#!/bin/bash

./run_heavinet.sh format voice.wav 16
./run_heavinet.sh load voice.wav voice.wav 16

./run_heavinet.sh train voice.wav 16 100
./run_heavinet.sh predict voice.wav voice.wav 16

./run_heavinet.sh train voice.wav 16 100
./run_heavinet.sh predict voice.wav voice.wav 16

./run_heavinet.sh train voice.wav 16 100
./run_heavinet.sh predict voice.wav voice.wav 16
