% downsample with default 'fir' filter of n=30
% sampling frequency is equal to  origianl_fx/2^(9-bits)
% inversevly proportionate to precision of data
function [y] = down_sample(x, factor)
y = x;
    for i = 1:factor
        y = decimate(y,2,'fir');
    end
end

