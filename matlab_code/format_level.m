
function [format_iterable] = format_level(iterable, receptive_field)

index = make_index(receptive_field);


len_iterable = length(iterable);
len_index = length(index);

format_iterable = zeros(len_iterable, len_index);
if receptive_field == 0
    limit = 1;
else
    limit = 2^(receptive_field-1)+1;
end

if limit > len_iterable
    error_msg = 'Error: The length of the receptive field was too large for the song'
    error(error_msg);
end

for i = 0: limit - 1
    format_iterable(i+1,:) = iterable( mod((i+index),len_iterable)+1 );
end

for i = limit:len_iterable-limit -1 
    format_iterable(i+1,:) = iterable( i + index );
end

for i = len_iterable-limit: len_iterable - 1
    format_iterable(i+1,:) = iterable( mod((i+index),len_iterable)+1 );
end

end

function [index] = make_index(receptive_field)

    loop=1;
    for i=1:receptive_field
        index(i) = -1*2^(receptive_field-i);
        loop = loop + 1;
    end
    index(loop)=0;
    loop=loop+1;
    for i=loop:loop+receptive_field-1
        index(i) = 2^(i-loop);
    end
    
end