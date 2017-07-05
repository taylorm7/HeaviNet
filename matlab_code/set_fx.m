function [fx, filter_fx] = set_fx(song_location, data_location, n_levels)

data_file = strcat(data_location, '/fx_info.mat');

song_info = audioinfo(song_location);
fx = song_info.SampleRate;
syms f positive
eqn = f.^(n_levels)==16000;
solf = solve(eqn,f);
filter_fx = double(solf(1));

save(data_file, 'fx', 'filter_fx');
end