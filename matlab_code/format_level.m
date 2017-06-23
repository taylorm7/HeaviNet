
function [format_input] = format_level(input)

A = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32];
index = [ -2 -1 0 1 2 ];
%index = [ -1 -0 1 2 3 ];

len_A = length(A);
len_index = length(index);

format_input = zeros(len_A, len_index);

for i = 0: len_index
    %indices = mod((i+index),len_A)+1
    format_input(i+1,:) = A ( mod((i+index),len_A)+1 )
end

for i = len_index:len_A-len_index
    i;
    %format_input(i,:) = A((i+index))
end

end