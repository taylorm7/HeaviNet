function [song, finished_song, original_fx] = audio_finish(data_location, n_levels, downsample_rate, receptive_field)
disp(data_location);

song = 0;
for i = 0:(n_levels-1)
    read_name = strcat('song_', int2str(i), '_r', int2str(receptive_field) );
    read_file = strcat(data_location, '/', read_name, '.mat');
    song_level = load(read_file);
    song = song + getfield(song_level, read_name);
end

moving_average = movmean(song, 30);
finished_song = smoothdata( song);
    
    
D=song-finished_song;
MSE=mean(D.^2);
fprintf('Difference after hampel and moving average filter = %g\n', MSE )

finish_file = strcat(data_location, '/song_r', int2str(receptive_field), '.wav');
[~, original_fx] = get_fx(data_location, 0);


s = (finished_song - (n_levels*128));
abs_max = max(abs(s));

finished_song = s/abs_max;

disp('Song saved at');
audiowrite(finish_file, finished_song, original_fx);
final_info = audioinfo(finish_file)
end

