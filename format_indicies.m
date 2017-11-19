function [index] = format_indicies(receptive_field, fx, passband_fx)

[index, limit, factor] = get_index(receptive_field, fx, passband_fx);
%disp(index);
[index] = re_order(receptive_field, index);
index = int32(index);
%disp(index);
end