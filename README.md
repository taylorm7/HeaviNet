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
