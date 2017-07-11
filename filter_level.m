function [] = filter_level(read_location, save_location, level, receptive_field,  data_location) 
    
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
    
    %filter here    
    %input_analog = digital_to_analog(input_digital, Q);
    %upsampled_analog = up_sample(input_analog, 1);
    %seed = analog_to_digital(upsampled_analog, Q);
    
    %downsampled_error = down_sample(upsampled_analog, 1);
    %D=input_analog-downsampled_error;
    %MSE=mean(D.^2);
    %fprintf('Level:%d upsample error between original and upsample = %g\n', level, MSE )

    seed = format_level(input_digital, receptive_field, fx, passband_fx);
    
    save(save_location, 'seed', '-v7.3');
end