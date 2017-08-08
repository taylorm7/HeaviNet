function [song, inputs, finished_song, original_fx] = audio_finish(data_location, n_levels, downsample_rate, receptive_field, seed_location)
disp(data_location);
bits = 8;
inputs = cell(n_levels,1);
song = 0;

for level = 1:n_levels
    read_location = strcat(seed_location, '/song_', int2str(level-1), '_r', int2str(receptive_field), '.mat' );
    disp(read_location);
    [passband_fx, fx] = get_fx(data_location, level);
    [index, limit, factor, ma_field, ha_field, ha_threshold] = get_index(receptive_field, fx, passband_fx);
    
    N = 2^(bits);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    input_digital = transpose(importdata(read_location));
    input_analog = digital_to_analog(input_digital, Q);
    input_signal = mu_inverse(input_analog, mu, Q);
    
    input_moving_average = movmean(input_signal, ma_field);
    input_hampel = hampel( input_moving_average, ha_field);
    
    inputs{level} = input_signal;
    song = song + input_signal;
    
    D=input_signal-input_hampel;
    MSE=mean(D.^2);
    fprintf('Level:%d Passband:%g Factor:%g Filter Field:%d\n', level, passband_fx, factor, ma_field);  
    %fprintf('Difference after hampel and moving average filter = %g\n', MSE )
    
    %figure()
    %plot(input_signal)
    %hold on
    %plot(input_moving_average)
    
    %neuralnet_song = strcat(data_location, '/out_', int2str(level-1), '_r', int2str(receptive_field), '_nn.wav');
    %filtered_song = strcat(data_location, '/out_', int2str(level-1), '_r', int2str(receptive_field), '.wav');
    
    %audiowrite(neuralnet_song, input_signal, fx);
    %audiowrite(filtered_song, input_moving_average, fx);
    %audioinfo(filtered_song)
end

finished_song = sgolayfilt(song,3,41);
    
D=song-finished_song;
MSE=mean(D.^2);
fprintf('Difference after filter = %g\n', MSE )

finish_file = strcat(seed_location, '/song_r', int2str(receptive_field), '.wav');
[~, original_fx] = get_fx(data_location, 0);

abs_max = max(abs(finished_song));
finished_song = finished_song/abs_max;

disp('Song saved at');
audiowrite(finish_file, finished_song, original_fx);
final_info = audioinfo(finish_file)
end

