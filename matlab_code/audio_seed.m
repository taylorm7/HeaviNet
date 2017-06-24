function [song, fx, seed] = audio_seed(level, song_location, data_location, downsample_rate, total_levels)
%song_location = '/home/sable/HeaviNet/data/songs/voice.wav';
%data_location = '/home/sable/HeaviNet/data/voice.wav.data';

disp(song_location);
disp(data_location);

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

[x_seed, x_fx_seed, y_seed, seed, z_seed] = create_level_downsample(bits, song, fx, total_levels);

save(data_location, 'level', 'seed');
end