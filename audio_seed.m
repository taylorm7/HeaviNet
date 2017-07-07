function [song, fx, seed, seed_signal] = audio_seed(level, song_location, data_file, downsample_rate, n_levels, receptive_field, data_location)
%song_location = '/home/sable/HeaviNet/data/songs/voice.wav';
%data_location = '/home/sable/HeaviNet/data/voice.wav.data';

disp(data_file);

level = level +1;
disp(song_location);

[song, fx ] = audio_read(song_location, downsample_rate);
[original_fx, filter_fx] = get_fx(data_location);

if original_fx ~= fx
    error_msg = 'Error: The frequency rate of the seed value and song value are incompatable'
    error(error_msg);
end

passband_fx = filter_fx^level
[seed, seed_signal] = create_filter(level, song, fx, passband_fx);
seed = format_level(seed, receptive_field, fx, passband_fx);

save(data_file, 'seed', '-v7.3');
end