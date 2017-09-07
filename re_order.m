function [iterable, indicies] = re_order(receptive_field, iterable)
clip_size = receptive_field + 1;
index = 1;
indicies = [];
while( index <= clip_size)
    indicies = [indicies index ];
    index = index + 2;
end
if mod(receptive_field,2) == 0
    index = index -3;
else
    index = index -1;
end
while(index >= 1)
    indicies = [indicies index];
    index = index -2;
end
iterable = iterable(:, indicies);
end
