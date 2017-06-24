function [song, fx, x_down, x_fx, y, inputs, input_z, targets, target_z, inputs_formatted] = audio_format(song_location, data_location, downsample_rate,n_levels, receptive_field)
disp(song_location);
disp(data_location);
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


x_down = cell(n_levels,1);
x_fx = cell(n_levels,1);
y = cell(n_levels,1);
inputs = cell(n_levels,1);
input_z = cell(n_levels,1);

inputs_formatted= cell(n_levels,1);

targets = cell(n_levels,1);
target_z = cell(n_levels,1);

for i = 1:n_levels
[x_down{i}, x_fx{i}, y{i}, inputs{i}, input_z{i}] = create_level_downsample(i, song, fx, n_levels);
[targets{i}, target_z{i}] = create_solution_downsample(i, song, fx, n_levels);
end

for i = 1:n_levels
inputs_formatted{i} = format_level(inputs{i}, receptive_field);
inputs_formatted{i} = int32(inputs_formatted{i});
end

disp("Song");
disp(length(song));
disp("Fx");
disp(fx);

save(data_location, 'n_levels', 'inputs', 'targets', 'inputs_formatted');
end
