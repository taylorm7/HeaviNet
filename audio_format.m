% audio_format: formats a wav file into a readable matlab input
% inputs: 
% song_location, string of input song to read
% data_location, string of data location to write matlab_song.mat
% downsample_rate, rate to downsample audio, typically run at 0
% n_levels, number of levels for heavinet, typically run at 8
% receptive_field, size of receptive field for sampling functions,
% typically run at 1
% training_data, True if training data, False if seed data
% seed_directory, string location of 
% outputs:
% matlab_seed.mat or matlab_song.mat file containing levels, level
% frequencies, and receptive field for each level
function [song, fx, inputs, inputs_signal, targets, targets_signal, inputs_formatted, levels] = audio_format(song_location, data_location, downsample_rate, n_levels, receptive_field, training_data, seed_directory)
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
% read audio, record the sampling frequency of original audio
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

levels = cell(n_levels,1);

indicies = cell(n_levels,1);
frequencies = cell(n_levels,1);
factors = cell(n_levels,1);

% filter data using hampel and sgolay
song =  hampel(song, 3, 0.5);
song = sgolayfilt(song,5,41);
song_max = max(abs(song));

for i = 1:n_levels
    % for each level, format the sampling matrix according to receptive
    % field
    passband_fx = get_fx(data_location, i);
    [indicies{i}, factors{i}] = format_indicies(receptive_field, fx, passband_fx);
    frequencies{i} = passband_fx;
    
    fprintf('Level:%d fx:%f\n', i, passband_fx);
end



fprintf('Song:%d fx:%g\n', length(song), fx);
% save output to matlab_song.mat file
save(data_file, 'fx', 'receptive_field', 'n_levels', 'indicies', 'frequencies', 'factors', 'song', '-v7.3');
end
