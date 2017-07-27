function [input_signal, input_moving_average] = filter_level(read_location, save_location, level, receptive_field, data_location) 
%filter_level('/home/sable/HeaviNet/data/beethoven_7.wav.data/song_26_r9.mat', '/home/sable/HeaviNet/data/beethoven_7.wav.data/test_out.wav', 26, 9, '/home/sable/HeaviNet/data/beethoven_7.wav.data');
    disp(read_location);
    disp(save_location);
    
    level = level +1
    
    [passband_fx, fx] = get_fx(data_location, level);
    
    N = 2^(8);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    input_digital = transpose(importdata(read_location));
    input_analog = digital_to_analog(input_digital, Q);
    input_signal = mu_inverse(input_analog, mu, Q);

    %D=input_analog-downsampled_error;
    %MSE=mean(D.^2);
    %fprintf('Level:%d upsample error between original and upsample = %g\n', level, MSE )

    seed = format_level(input_signal, receptive_field, fx, passband_fx);
    [index, limit, factor] = get_index(receptive_field, fx, passband_fx);
    
%     seed_smoothed = zeros( size(seed,1) , 1 );
%     for i=1:size(seed,1)
%         hampel_result = hampel( seed(i,:), receptive_field);
%         seed_smoothed(i) = hampel_result(receptive_field + 1);
%     end
    
    input_hampel = hampel( input_signal, ceil(factor/2) );
    input_moving_average = movmean(input_hampel,ceil(factor/2));
    
    figure()
    plot(input_signal)
    hold on
    plot(input_moving_average)
    
    %save(save_location, 'seed', '-v7.3');
    
    neuralnet_song = strcat(data_location, '/out_', int2str(level-1), '_r', int2str(receptive_field), '_nn.wav');
    filtered_song = strcat(data_location, '/out_', int2str(level-1), '_r', int2str(receptive_field), '.wav');
    audiowrite(neuralnet_song, input_signal, fx);
    audiowrite(filtered_song, input_moving_average, fx);
end