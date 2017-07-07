function [fx, filter_fx] = get_fx(data_location)
data_file = strcat(data_location, '/fx_info.mat');
fxs = load(data_file);

fx = fxs.fx;
filter_fx = fxs.filter_fx;
end