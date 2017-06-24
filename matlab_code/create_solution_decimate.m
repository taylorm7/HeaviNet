function [y_digital,y] = create_solution_decimate(level, song, fx, total_levels)

    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(level + 1);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    fx_factor = total_levels - level;
    x_fx = fx / 2^(fx_factor);
    
    fprintf('Solution Decimate Level:%d bits:%d fx:%g N:%d mu:%d Q:%g \n', level-1, level,x_fx, N,mu, Q);
    
    % downsample and upsample based on bit level 
    % (higher precision, lower downsampling)
    
    x_down = down_sample(song, fx_factor);
    x_up = up_sample(x_down, fx_factor);
    
    % compute error for downsampling and upsampling
    D=x_up-song;
    MSE=mean(D.^2);
    fprintf('Error between original and filtered = %g\n',MSE )

    % perform mu-law transform and digitize compressed data
    y_nonlinear = mu_trasform(x_up, mu, Q);
    y_digital = analog_to_digital(y_nonlinear, Q);
    % compute analog to digital, and perform inverse mu-law transform
    y_analog = digital_to_analog(y_digital, Q);
    y = mu_inverse(y_analog, mu, Q);
    
    % compute error for mu-law transform
    D=x_up-y;
    MSE=mean(D.^2);
    fprintf('Error between filtered and quantized = %g\n',MSE )
    
    % compute final error for companded and digitized data
    D=y-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
    %plot and write data to _.mat file
    %plot_q(x_down, y);

end