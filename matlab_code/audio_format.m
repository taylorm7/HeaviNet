function [song, fx, inputs, inputs_signal, targets, targets_signal, inputs_formatted] = audio_format(song_location, data_location, downsample_rate,n_levels, receptive_field)
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

clipped_length =floor(length(song)/2^(n_levels+downsample_rate))*2^(n_levels+downsample_rate);
song = song(1:clipped_length);
[song] = down_sample(song, downsample_rate);
fx = fx/2^downsample_rate;

inputs = cell(n_levels,1);
inputs_signal = cell(n_levels,1);

inputs_formatted= cell(n_levels,1);

targets = cell(n_levels,1);
targets_signal = cell(n_levels,1);

for i = 1:n_levels
[inputs{i}, inputs_signal{i}] = create_level_decimate(i, song, fx, n_levels);
[targets{i}, targets_signal{i}] = create_solution_decimate(i, song, fx, n_levels);
end

for i = 1:n_levels
inputs_formatted{i} = format_level(inputs{i}, receptive_field);
inputs_formatted{i} = int32(inputs_formatted{i});
end

disp("Song");
disp(length(song));
disp("Fx");
disp(fx);

save(data_file, 'receptive_field', 'n_levels', 'inputs_formatted', 'targets');
end
