function [index, limit, factor] = get_index(receptive_field, fx, passband_fx)

if passband_fx > fx/2
    % nofilter
    disp('No filter for this level');
    factor = 1;
else
    factor = fx / passband_fx;
end
index = make_index(receptive_field, factor);
limit = round(receptive_field*factor);
end

function [index] = make_index(receptive_field, factor)
    loop=1;
    index = zeros(1, 2*receptive_field + 1);
    for i=1:receptive_field
        index(i) = -1*round((receptive_field-i+1)*factor);
        loop = loop + 1;
    end
    index(loop)=0;
    loop=loop+1;
    for i=loop:loop+receptive_field-1
        index(i) = round((i-loop+1)*factor);
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