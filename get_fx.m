function [passband_fx, fx, filter_fx] = get_fx(data_location, level)
data_file = strcat(data_location, '/fx_info.mat');
fxs = load(data_file);

fx = fxs.fx;
filter_fx = fxs.filter_fx;

passband_fx = filter_fx^level;
end