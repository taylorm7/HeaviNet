% get_index: find indexing value according to level passband frequency
% inputs:
% receptive_field, integer of levels receptive field
% fx, sampling rate of audio
% passband_fx, passband frequency of given level
% outputs:
% index, integer index of used to format audio
% limit, integer value for max necessary size of audio
% ma_field, double value for movang average field
% ha_field, double value for hampel filter field
% ha_threshol, double value for hampel filter threshold
function [index, limit, factor, ma_field, ha_field, ha_threshold] = get_index(receptive_field, fx, passband_fx)

if passband_fx > fx/2
    % no filter
    disp('No filter for this level');
    factor = 1;
else
    factor = fx / passband_fx;
end
index = make_index(receptive_field, factor);
limit = round(receptive_field*factor);
ma_field = 2.*round((factor+1)/2)-1;
ha_field = 3;
ha_threshold = 1;
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
end