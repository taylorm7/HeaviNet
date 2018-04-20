function [y_digital, y, x_max, x] = create_filter(level, song, fx, passband_fx, receptive_field, data_location, bits)
    
    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(bits);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    [index, limit, factor, ma_field, ha_field, ha_threshold] = get_index(receptive_field, fx, passband_fx);
    passband_ripple = 0.2;
    passband_limit = get_fx(data_location, level+1);

    fprintf('bits:%d fx:%d passband:%g passband limit:%g N:%d mu:%d Q:%g \n', ...
        bits, fx, passband_fx, passband_limit, N, mu, Q);

    if passband_fx > fx/2
        % nofilter
        disp('No filter for this level');
        x = song;
    else

        lowpass_iir = designfilt('lowpassiir','FilterOrder',8, ...
                'PassbandFrequency',passband_fx,'PassbandRipple', passband_ripple, ...
                'SampleRate',fx);
        
        x = filter(lowpass_iir, song);
        x =  hampel(x, 3, 0.5);
        x = sgolayfilt(x,5,41);

        
    end
    y_nonlinear = mu_trasform(x, mu, Q);

    
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