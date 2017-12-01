function [song, fx, inputs, inputs_signal, targets, targets_signal, inputs_formatted, levels] = audio_format(song_location, data_location, downsample_rate, n_levels, receptive_field, training_data, seed_directory)
%[song, fx, i, inputs_signal, targets, targets_signal, inputs_formatted] = audio_format('/home/sable/HeaviNet/data/songs/beethoven_7.wav', '/home/sable/HeaviNet/data/beethoven_7.wav.data', 0, 8, 1, 1);
disp(song_location);
if training_data == 1
    data_file = strcat(data_location, '/matlab_song_r', int2str(receptive_field), '.mat');
else
    data_file = strcat(seed_directory, '/matlab_seed_r', int2str(receptive_field), '.mat');
end
disp(data_file);
disp('Receptive Field');
disp(receptive_field);
disp('Downsample');
disp(downsample_rate);
disp('Levels');
disp(n_levels);

[song, fx ] = audio_read(song_location, downsample_rate);
if training_data == 1
    [filter_fx] = set_fx(fx, data_location, n_levels);
else
    [~, original_fx] = get_fx(data_location, 1);
    if original_fx ~= fx
        error_msg = 'Error: The frequency rate of the seed value and song value are incompatable'
        error(error_msg);
    end
end
inputs = cell(n_levels,1);
inputs_signal = cell(n_levels,1);

% inputs_formatted= cell(n_levels,1);
levels = cell(n_levels,1);
%targets = cell(n_levels,1);
%targets_signal = cell(n_levels,1);

indicies = cell(n_levels,1);
frequencies = cell(n_levels,1);
factors = cell(n_levels,1);


song =  hampel(song, 3, 0.5);
song = sgolayfilt(song,5,41);
%song_max = max(abs(song));
%song = song./song_max;

for i = 1:n_levels
    
    passband_fx = get_fx(data_location, i);
    [indicies{i}, factors{i}] = format_indicies(receptive_field, fx, passband_fx);
    frequencies{i} = passband_fx;
    
    fprintf('Level:%d fx:%f\n', i, passband_fx);
    
    %[inputs{i}, inputs_signal{i}] = create_filter(i, song, fx, passband_fx, receptive_field, data_location, 8);
    %targets{i} = inputs{i};
    %fprintf('Solution:%d\n', i);
    %passband_fx = get_fx(data_location, i+1);
    %song_target = circshift(song, -1);
    %[targets{i}, targets_signal{i}, levels{i}] = create_filter(i+1, song, fx, passband_fx, 0, data_location, 8);
    
    %if training_data == 1
    %    signal_location = strcat(data_location, '/signal_', int2str(i), '.wav');
    %else
    %    signal_location = strcat(seed_directory, '/seed_', int2str(i), '.wav');
    %end
    %audiowrite(signal_location, inputs_signal{i}, fx);
end

%for i = 1:n_levels
%fprintf('Formatting level:%d\n', i);
%passband_fx = get_fx(data_location, i);
%[inputs_formatted{i}] = format_level(inputs{i}, receptive_field, fx, passband_fx);
%end

fprintf('Song:%d fx:%g\n', length(song), fx);

%save(data_file, 'receptive_field', 'n_levels', 'inputs_formatted', 'targets', 'indicies', 'frequencies', 'song', '-v7.3');
save(data_file, 'fx', 'receptive_field', 'n_levels', 'indicies', 'frequencies', 'factors', 'song', '-v7.3');
end
