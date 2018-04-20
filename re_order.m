% re_order: reorder index matrix to center values
% inputs:
% receptive_field, integer of levels receptive field
% iterable, the audio matrix to be sampled
% outputs:
% iterable, matrix of the audio to be sampled
% indices, matrix of re_ordered indicies
function [iterable, indicies] = re_order(receptive_field, iterable)
clip_size = receptive_field;

index = 1;
indicies = [];
while( index <= clip_size)
    indicies = [indicies index ];
    index = index + 2;
end
if mod(receptive_field,2) == 1
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
