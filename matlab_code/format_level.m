
function [format_iterable, index] = format_level(iterable, receptive_field, level, n_levels)

index = make_index(receptive_field, level, n_levels);


len_iterable = length(iterable);
len_index = length(index);

format_iterable = zeros(len_iterable, len_index);


limit = (receptive_field)*2^(n_levels-level);

if limit >= len_iterable
    error_msg = 'Error: The length of the receptive field was too large for the song'
    error(error_msg);
end

parfor i = 0: limit - 1
    format_iterable(i+1,:) = iterable( mod((i+index),len_iterable)+1 );
end

parfor i = limit:len_iterable-limit -1 
    format_iterable(i+1,:) = iterable( i + index + 1 );
end

parfor i = len_iterable-limit: len_iterable - 1
    format_iterable(i+1,:) = iterable( mod((i+index),len_iterable)+1 );
end

end

function [index] = make_index(receptive_field, level, n_levels)
    loop=1;
    index = zeros(1, 2*receptive_field + 1);
    for i=1:receptive_field
        index(i) = -1*(receptive_field-i+1)*2^(n_levels-level);
        loop = loop + 1;
    end
    index(loop)=0;
    loop=loop+1;
    for i=loop:loop+receptive_field-1
        index(i) = (i-loop+1)*2^(n_levels-level);
    end
    
%     loop=1;
%     for i=1:receptive_field
%         index(i) = -1*2^(receptive_field-i);
%         loop = loop + 1;
%     end
%     index(loop)=0;
%     loop=loop+1;
%     for i=loop:loop+receptive_field-1
%         index(i) = 2^(i-loop);
%     end
    
end