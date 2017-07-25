function [] = filter_level(read_location, save_location, level, receptive_field, data_location) 
%filter_level('/home/sable/HeaviNet/data/beethoven_7.wav.data/song_26_r9.mat', '/home/sable/HeaviNet/data/beethoven_7.wav.data/test_out.m', 26, 9, '/home/sable/HeaviNet/data/beethoven_7.wav.data');
    disp(read_location);
    disp(save_location);
    
    level = level +1
    
    [passband_fx, fx] = get_fx(data_location, level);
    
    N = 2^(level);
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
    
    seed_smoothed = zeros( size(seed,1) , 1 );
    for i=1:size(seed,1)
        hampel_result = hampel( seed(i,:), receptive_field);
        seed_smoothed(i) = hampel_result(receptive_field + 1);
    end
    
    figure()
    plot(input_signal)
    hold on
    plot(seed_smoothed)
    
    
    save(save_location, 'seed', '-v7.3');
end