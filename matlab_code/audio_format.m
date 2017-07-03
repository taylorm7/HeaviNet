function [song, fx, inputs, inputs_signal, targets, targets_signal, inputs_formatted] = audio_format(song_location, data_location, downsample_rate, n_levels, receptive_field)
disp(song_location);
data_file = strcat(data_location, '/matlab_song_r', int2str(receptive_field), '.mat');
disp(data_file);
disp("Receptive Field");
disp(receptive_field);
disp("Downsample");
disp(downsample_rate);
disp("Levels");
disp(n_levels);

[song, fx] = audioread(song_location);
song_info = audioinfo(song_location);
if song_info.NumChannels == 2
    song = (song(:,1) + song(:,2))/2;
end

if downsample_rate > 0
    clipped_length =floor(length(song)/downsample_rate)*downsample_rate;
    song = song(1:clipped_length);
    song = decimate(song,downsample_rate,'fir');
    fx = fx/downsample_rate;
end

syms f positive
eqn = f.^(n_levels)==16000;
solf = solve(eqn,f);
filter_fx = double(solf(1))

inputs = cell(n_levels,1);
inputs_signal = cell(n_levels,1);

inputs_formatted= cell(n_levels,1);

targets = cell(n_levels,1);
targets_signal = cell(n_levels,1);

for i = 1:n_levels
    fprintf('Level:%d\n', i);
    passband_fx = filter_fx^i;
    [inputs{i}, inputs_signal{i}] = create_filter(i, song, fx, passband_fx);
    fprintf('Solution:%d\n', i);
    [targets{i}, targets_signal{i}] = create_filter(i+1, song, fx, passband_fx);
end

% job = batch('format_level', 1 , {inputs{1}, receptive_field, 1, n_levels}, 'pool', 2);
% wait(job);
% diary(job);
% test_job = fetchOutputs(job);
% delete(job);

for i = 1:n_levels
fprintf('Formatting level:%d\n', i);
passband_fx = filter_fx^i;
inputs_formatted{i} = format_level(inputs{i}, receptive_field, fx, passband_fx);
inputs_formatted{i} = int32(inputs_formatted{i});
end

fprintf('Song:%d fx:%g\n', length(song), fx);

save(data_file, 'receptive_field', 'n_levels', 'inputs_formatted', 'targets', '-v7.3');
end
