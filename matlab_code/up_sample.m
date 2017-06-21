% upsample with 'fir' filter with window size 8
% sampling frequency is equal to  origianl_fx/2^(9-bits)
% inversevly proportionate to precision of data
function [z] = up_sample(y, factor)
z = y;
    for i = 1: factor
        z = interp(z,2, 8);
    end
end
