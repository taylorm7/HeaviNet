function [format_iterable] = format_level(iterable, receptive_field, fx, passband_fx)

[index, limit, factor] = get_index(receptive_field, fx, passband_fx);
disp(index);

len_iterable = length(iterable);
len_index = length(index);

format_iterable = zeros(len_iterable, len_index);

if limit >= len_iterable
    error_msg = 'Error: The length of the receptive field was too large for the song'
    error(error_msg);
end

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
