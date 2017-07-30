function [y_digital, y, x] = create_filter(level, song, fx, passband_fx, receptive_field, data_location)
    
    % set values for mu transform, assuming standard -1,1 audio
    bits = 8;
    N = 2^(bits);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    [index, limit, factor, ma_field, ha_field, ha_threshold] = get_index(receptive_field, fx, passband_fx);
    passband_ripple = 0.2;
    passband_limit = get_fx(data_location, level+1);
    %fprintf('bits:%d fx:%d passband:%g passband ripple:%g N:%d mu:%d Q:%g \n', ...
    %    level, fx, passband_fx, passband_ripple, N, mu, Q);
    fprintf('bits:%d fx:%d passband:%g passband limit:%g N:%d mu:%d Q:%g \n', ...
        bits, fx, passband_fx, passband_limit, N, mu, Q);

    if passband_fx > fx/2
        %# nofilter
        disp('No filter for this level');
        x = song;
    else
        %iir_filter = designfilt('lowpassiir','FilterOrder',8, ...
        %     'PassbandFrequency',passband_fx,'PassbandRipple', passband_ripple, ...
        %     'SampleRate',fx);

        iir_filter = designfilt('bandpassiir','FilterOrder',20, ...
               'HalfPowerFrequency1',passband_fx,'HalfPowerFrequency2', passband_limit, ...
               'SampleRate',fx);

        x = filter(iir_filter, song);

        
        
        %plot fir filter
        %fvtool(lowpass_filter)
        
    end
    % compute error for filtering
    D=x-song;
    MSE=mean(D.^2);
    fprintf('Error between original and filtered = %g\n',MSE )
    
    x_moving_average = movmean(x, ma_field);
    x_hampel = medfilt1( x_moving_average, ha_field);
    
    
    D=x-x_moving_average;
    MSE=mean(D.^2);
    fprintf('Difference after hampel and moving average filter = %g\n', MSE )
    
    % perform mu-law transform and digitize compressed data
    %y_nonlinear = mu_trasform(x, mu, Q);
    y_nonlinear = mu_trasform(x_hampel, mu, Q);
    
    y_digital = analog_to_digital(y_nonlinear, Q);
    % compute analog to digital, and perform inverse mu-law transform
    y_analog = digital_to_analog(y_digital, Q);
    y = mu_inverse(y_analog, mu, Q);
    
    % compute error for mu-law transform
    D=x-y;
    MSE=mean(D.^2);
    fprintf('Error between filtered and quantized = %g\n',MSE )
    
    % compute final error for companded and digitized data
    D=y-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
    %plot original and filtered/companded
    %plot_q(song, y);

end