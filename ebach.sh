#!/bin/bash

./run_heavinet.sh format beethoven_7.wav 16
./run_heavinet.sh load beethoven_7.wav beethoven_7.wav 16

./run_heavinet.sh train beethoven_7.wav 16 1000
./run_heavinet.sh generate beethoven_7.wav beethoven_7.wav 16

./run_heavinet.sh train beethoven_7.wav 16 1000
./run_heavinet.sh generate beethoven_7.wav beethoven_7.wav 16
