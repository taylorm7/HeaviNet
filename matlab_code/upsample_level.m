function [] = upsample_level(read_location, save_location, level ) 
    
    disp(read_location);
    disp(save_location);
    
    N = 2^(level);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    input_digital = transpose(importdata(read_location));
    input_analog = digital_to_analog(input_digital, Q);
    
    upsampled_analog = up_sample(input_analog, 1);
    seed = analog_to_digital(upsampled_analog, Q);
    
    downsampled_error = down_sample(upsampled_analog, 1);
    D=input_analog-downsampled_error;
    MSE=mean(D.^2);
    fprintf('Level:%d upsample error between original and upsample = %g\n', level, MSE )
    
    save(save_location, 'seed');
end

%function [] = test_file(name, value)
%    fprintf("%s %d\n",name, value);
%end
