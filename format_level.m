% format_level: sample song according to given receptive field and passband
% fx
% inputs:
% iterable, the audio matrix to be sampled
% receptive_field, receptive field of given level
% fx, sampling rate of audio
% passband_fx, passband frequency of given level
% outputs:
% format_iterable, formatted audio matrix according to receptive field and
% passband fx
% index, matrix of formatted indexs according to level receptive field

function [format_iterable, index] = format_level(iterable, receptive_field, fx, passband_fx)

% create the index matrix and reorder to centered samples
[index, limit, factor] = get_index(receptive_field, fx, passband_fx);
[index] = re_order(receptive_field, index);


len_iterable = length(iterable);
len_index = length(index);

format_iterable = zeros(len_iterable, len_index);
return;

if limit >= len_iterable
    error_msg = 'Error: The length of the receptive field was too large for the song'
    error(error_msg);
end

% sample the audio matrix and store in the formatted matrix
for i = 0: limit - 1
    format_iterable(i+1,:) = iterable( mod((i+index),len_iterable)+1 );
end

for i = limit:len_iterable-limit -1 
    format_iterable(i+1,:) = iterable( i + index + 1 );
end

for i = len_iterable-limit: len_iterable - 1
    format_iterable(i+1,:) = iterable( mod((i+index),len_iterable)+1 );
end

format_iterable = int32(format_iterable);
end
