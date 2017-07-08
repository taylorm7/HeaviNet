function [filter_fx] = set_fx(fx, data_location, n_levels)

data_file = strcat(data_location, '/fx_info.mat');

syms f positive
eqn = f.^(n_levels)==7350;
solf = solve(eqn,f);
filter_fx = double(solf(1));

save(data_file, 'fx', 'filter_fx');
end