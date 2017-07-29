function [song, fx, seed, seed_signal] = audio_seed(level, song_location, data_file, downsample_rate, n_levels, receptive_field, data_location)
%song_location = '/home/sable/HeaviNet/data/songs/voice.wav';
%data_location = '/home/sable/HeaviNet/data/voice.wav.data';

disp(data_file);

level = level +1;
disp(song_location);

[song, fx] = audio_read(song_location, downsample_rate);
[passband_fx, original_fx] = get_fx(data_location, level);

if original_fx ~= fx
    error_msg = 'Error: The frequency rate of the seed value and song value are incompatable'
    error(error_msg);
end

[seed, seed_signal] = create_filter(8, song, fx, passband_fx, receptive_field);
seed = format_level(seed, receptive_field, fx, passband_fx);

save(data_file, 'seed', '-v7.3');

seed_file = strcat(data_location, '/seed_', int2str(level-1), '_r', int2str(receptive_field), '.wav');
audiowrite(seed_file, seed_signal, fx);

end