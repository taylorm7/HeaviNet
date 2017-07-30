function [input_signal, input_moving_average] = filter_level(read_location, save_location, level, receptive_field, data_location) 
%filter_level('/home/sable/HeaviNet/data/beethoven_7.wav.data/song_26_r9.mat', '/home/sable/HeaviNet/data/beethoven_7.wav.data/test_out.wav', 26, 9, '/home/sable/HeaviNet/data/beethoven_7.wav.data');
    disp(read_location);
    disp(save_location);
    
    level = level +1
    
    [passband_fx, fx] = get_fx(data_location, level);
    [index, limit, factor, ma_field, ha_field, ha_threshold] = get_index(receptive_field, fx, passband_fx);
    
    N = 2^(8);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    input_digital = transpose(importdata(read_location));
    input_analog = digital_to_analog(input_digital, Q);
    input_signal = mu_inverse(input_analog, mu, Q);
    
    input_moving_average = movmean(input_signal, ma_field);
    input_hampel = medfilt1( input_moving_average, ha_field);
    
    
    D=input_signal-input_moving_average;
    MSE=mean(D.^2);
    fprintf('Level:%d Passband:%g Factor:%g Filter Field:%d\n', level, passband_fx, factor, ma_field);  
    fprintf('Difference after hampel and moving average filter = %g\n', MSE )
    
    %figure()
    %plot(input_signal)
    %hold on
    %plot(input_moving_average)
    
    neuralnet_song = strcat(data_location, '/out_', int2str(level-1), '_r', int2str(receptive_field), '_nn.wav');
    filtered_song = strcat(data_location, '/out_', int2str(level-1), '_r', int2str(receptive_field), '.wav');
    
    audiowrite(neuralnet_song, input_signal, fx);
    audiowrite(filtered_song, input_moving_average, fx);
    audioinfo(filtered_song)
    
    seed_nonlinear = mu_trasform(input_hampel, mu, Q);
    seed_digital = analog_to_digital(seed_nonlinear, Q);
    
    seed = format_level(seed_digital, receptive_field, fx, passband_fx);
    save(save_location, 'seed', '-v7.3');
    
end
