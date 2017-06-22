function [] = upsample_level(data_location, song_name)
    song_file = strcat( data_location, "/", song_name , ".mat");
    
    disp(song_file);
    disp(song_name);
    
    song = transpose(importdata(song_file));
    
end

%function [] = test_file(name, value)
%    fprintf("%s %d\n",name, value);
%end
