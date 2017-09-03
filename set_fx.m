function [filter_fx] = set_fx(fx, data_location, n_levels)

data_file = strcat(data_location, '/fx_info.mat');
fx_offset = 2;
fx_order = 3;
use_exponential = 0;
fx_target = 16000;
syms f positive
  
if use_exponential
    %exponential function for filter changes
    eqn = f.^(n_levels+1)+fx_offset==fx_target;
else
    %polynomial function of order fx_order
    eqn = ((n_levels)*f).^(fx_order)+fx_offset==fx_target;
end
solf = solve(eqn,f);
filter_fx = double(solf(1));

if use_exponential
    fprintf('Offset Frequency:%d Target Frequency:%d Exponential Function:%g^level+%d\n', ...
        fx_offset, fx_target, filter_fx, fx_offset);
else
    fprintf('Offset Frequency:%d Target Frequency:%d Polynomial Function:(%g*level)^%d+%d\n', ...
        fx_offset, fx_target, filter_fx, fx_order, fx_offset);
end
save(data_file, 'fx', 'filter_fx','fx_offset', 'fx_order', 'use_exponential');
end
