function [y_digital, y, x] = create_filter(level, song, fx, passband_fx)
    
    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(level);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    passband_ripple = 0.2;
 
    fprintf('bits:%d fx:%d passband:%g passband ripple:%g N:%d mu:%d Q:%g \n', ...
        level, fx, passband_fx, passband_ripple, N, mu, Q);
    
    if passband_fx > fx/2
        %# nofilter
        disp('No filter for this level');
        x = song;
    else
        lowpass_filter = designfilt('lowpassiir','FilterOrder',8, ...
             'PassbandFrequency',passband_fx,'PassbandRipple', passband_ripple, ...
             'SampleRate',fx);
        %plot fir filter
        %fvtool(lowpass_filter)
        x = filter(lowpass_filter, song);
    end
    % compute error for filtering
    D=x-song;
    MSE=mean(D.^2);
    fprintf('Error between original and filtered = %g\n',MSE )

    % perform mu-law transform and digitize compressed data
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