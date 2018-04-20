% get_fx: read frequency rate for given heavinet run
% inputs:
% data_location, string of data location to read from
% level, integer of given level in run
% outputs:
% passband_fx, passband frequency of given level
% fx, sampling rate of audio
% filter_fx, filter frequency of given level
function [passband_fx, fx, filter_fx] = get_fx(data_location, level)
data_file = strcat(data_location, '/fx_info.mat');
fxs = load(data_file);

fx = fxs.fx;
filter_fx = fxs.filter_fx;


if fxs.use_exponential
    % exponential function for filter changes
    passband_fx = filter_fx^(level-1)+fxs.fx_offset;
else
    % polynomial function of order fx_order, value set in set_fx.m 
    passband_fx = ((level-1)*filter_fx)^(fxs.fx_order)+fxs.fx_offset;
end

end