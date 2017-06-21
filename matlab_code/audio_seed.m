function [song, fx] = audio_seed(level, song_location, data_location, downsample_rate, total_levels)
song_location = '/home/sable/HeaviNet/data/songs/voice.wav';
data_location = '/home/sable/HeaviNet/data/voice.wav.data';

[song, fx] = audioread(song_location);
song_info = audioinfo(song_location);
if song_info.NumChannels == 2
    song = (song(:,1) + song(:,2))/2;
end

bits = level + 1;
clipped_length =floor(length(song)/2^(total_levels+downsample_rate))*2^(total_levels+downsample_rate);
song = song(1:clipped_length);
[song] = down_sample(song, downsample_rate);
fx = fx/2^downsample_rate;
end