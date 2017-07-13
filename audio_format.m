function [song, fx, inputs, inputs_signal, targets, targets_signal, inputs_formatted] = audio_format(song_location, data_location, downsample_rate, n_levels, receptive_field)
disp(song_location);
data_file = strcat(data_location, '/matlab_song_r', int2str(receptive_field), '.mat');
disp(data_file);
disp('Receptive Field');
disp(receptive_field);
disp('Downsample');
disp(downsample_rate);
disp('Levels');
disp(n_levels);

[song, fx ] = audio_read(song_location, downsample_rate);

[filter_fx] = set_fx(fx, data_location, n_levels);

inputs = cell(n_levels,1);
inputs_signal = cell(n_levels,1);

inputs_formatted= cell(n_levels,1);

targets = cell(n_levels,1);
targets_signal = cell(n_levels,1);

for i = 1:n_levels
    fprintf('Level:%d\n', i);
    passband_fx = get_fx(data_location, i);
    [inputs{i}, inputs_signal{i}] = create_filter(8, song, fx, passband_fx);
    fprintf('Solution:%d\n', i);
    passband_fx = get_fx(data_location, i+1);
    [targets{i}, targets_signal{i}] = create_filter(8, song, fx, passband_fx);
end

% job = batch('format_level', 1 , {inputs{1}, receptive_field, 1, n_levels}, 'pool', 2);
% wait(job);
% diary(job);
% test_job = fetchOutputs(job);
% delete(job);

for i = 1:n_levels
fprintf('Formatting level:%d\n', i);
passband_fx = get_fx(data_location, i);
inputs_formatted{i} = format_level(inputs{i}, receptive_field, fx, passband_fx);
end

fprintf('Song:%d fx:%g\n', length(song), fx);

save(data_file, 'receptive_field', 'n_levels', 'inputs_formatted', 'targets', '-v7.3');
end
