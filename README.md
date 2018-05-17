# HeaviNet
Music Generation Neural Network

HeaviNet is a hierarchy of layered adaptive filterers realized via levels of 
auto-regressive convolution neural networks. The project is currently aimed 
towards learned filtering, learned compression, learned reconstruction, and 
generation of raw audio files. HeaviNet is named after the Heaviside function; 
and each level is most essentially a Heaviside function whose input is a sample 
of the previous level with resolution 2^(level-1) and output of resolution 
2^(level). The system takes as input a piece of audio (music or speech), 
and breaks down the sample into levels according to a frequency 
(Hz via lowpass filtering) and resolution (bits via mu law companding and 
quantization).  Each levels input is fed into a convolutional neural network, 
and trained according to the next levels output. In this way, HeaviNet can 
learn the most fundamental low frequency patterns/beats, the mid range 
melodies, and the high frequency harmonies of any audio sample.  
After training, the goal of HeaviNet is to realize adaptive and dynamic signal 
reconstruction, digital compression, and signal filtering; as well as speech 
and music generation.


This is a project developed by Taylor McGough advised by JoAnn Paul for 
a Master of Science Thesis in the Computer Engineering Department 
of Virginia Tech. 

For questions please contact TaylorM7@vt.edu

Publication is forthcoming. If you use any code or information, please reference

McGough, Taylor, and JoAnn Paul. "HeaviNet: Music Generation", (2017), Github Repository. https://github.com/taylorm7/HeaviNet

@misc{McGough2017,
  author = {McGough, Taylor and Paul, JoAnn},
  title = {HeaviNet: Music Generation},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/taylorm7/HeaviNet}}
}

Usage:

# formats the audio and creates directory
./run_heavinet.sh format bach_10.wav [receptive_field]

# formats the seed for generation, can be audio or random white noise
./run_heavinet.sh load bach_10.wav rand.wav [receptive_field]

# trains according to number of epochs
./run_heavinet.sh train bach_10.wav [receptive_field] [epochs]

# generates using the seed value supplied, 
./run_heavinet.sh generate bach_10.wav rand.wav [receptive_field]
output goes into /data/bach_10.wav.data/rand.wav.data/...

# parameters
Number of Levels is specified in run_heavinet 
LEVELS=8

Frequencies to filter are specified in set_fx.m
fx_offset = 25; $ frequency to start filtering at 
fx_order = 3; $ order of polynomial function for frequncies, x^1 specifies linear frequencies 
use_exponential = 1; $ using 2^x for frequencies; 1 uses exponential and 0 uses polynomial (order with fx_order)
fx_target = 8000; $ highest frequency to filter

When using branch 'master', i.e. using the WaveNet model, receptive field should be [1]. It's used in the sampling methods in the audio_format.m and can sample at the corresponding level frequencies. 

Let me know if you have any questions!
