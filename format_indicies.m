% format_indicies: format indiciies for level according to receptive field
% inputs:
% receptive_field, receptive field of given level
% fx, sampling rate of audio
% passband_fx, passband frequency of given level
% outputs:
% index, matrix of formatted indexs according to level receptive field
% factor, change factor for index samples
function [index, factor] = format_indicies(receptive_field, fx, passband_fx)
[index, limit, factor] = get_index(receptive_field, fx, passband_fx);
index = int32(index);
end