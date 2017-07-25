function [filter_fx] = set_fx(fx, data_location, n_levels)

data_file = strcat(data_location, '/fx_info.mat');
fx_offset = 50;
fx_order = 2;
fx_target = 8000;
syms f positive
  %exponential function for filter changes
%eqn = f.^(n_levels)+fx_offset==fx_target;
  %polynomial function of order fx_order,
eqn = ((n_levels)*f).^(fx_order)+fx_offset==fx_target;
solf = solve(eqn,f);
filter_fx = double(solf(1));

fprintf('Offset Frequency:%d Target Frequency:%d Filter Function:%g^%d+%d\n', ...
        fx_offset, fx_target, filter_fx, fx_order, fx_offset);

save(data_file, 'fx', 'filter_fx','fx_offset', 'fx_order');
end