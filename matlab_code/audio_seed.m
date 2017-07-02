function [song, fx, seed, seed_signal] = audio_seed(level, song_location, data_location, downsample_rate, n_levels, receptive_field)
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
clipped_length =floor(length(song)/2^(n_levels+downsample_rate))*2^(n_levels+downsample_rate);
song = song(1:clipped_length);
[song] = down_sample(song, downsample_rate);
fx = fx/2^downsample_rate;

[seed, seed_signal] = create_level_decimate(bits, song, fx, n_levels);
seed = format_level(seed, receptive_field, level, n_levels);

save(data_location, 'level', 'seed', '-v7.3');
end