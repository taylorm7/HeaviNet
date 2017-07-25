function [passband_fx, fx, filter_fx] = get_fx(data_location, level)
data_file = strcat(data_location, '/fx_info.mat');
fxs = load(data_file);

fx = fxs.fx;
filter_fx = fxs.filter_fx;

% exponential function for filter changes
passband_fx = filter_fx^(level)+fxs.fx_offset;

% polynomial function of order fx_order, value set in in set_fx.m 
%passband_fx = ((level-1)*filter_fx)^(fxs.fx_order)+fxs.fx_offset;

end