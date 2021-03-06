function [y_digital,y, x] = create_solution_filter(level, song, fx, total_levels)

    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(level + 1);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
   
    % f(x) = 2/pi * arctan( pi/2 * x * k) -> [0,1] for all x with slope of k
    passband_fx = 2/pi* atan(pi/2* (level/total_levels) * level/total_levels*( (level+1) /total_levels));
    stopband_fx = 2/pi* atan(pi/2* (level/total_levels) * (level+1*0.5)/total_levels*( (level+1) /total_levels));

    fprintf('Filter Solution Level:%d bits:%d passband:%g stopband:%g N:%d mu:%d Q:%g \n', ...
        level-1, level,passband_fx, stopband_fx, N,mu, Q);


    if level == total_levels
        %#nofilter
        x = song;
    else
        lowpass_filter = designfilt('lowpassfir', ...
            'PassbandFrequency', passband_fx,'StopbandFrequency', stopband_fx, ...
            'PassbandRipple',0.5,'StopbandAttenuation',65,'DesignMethod','kaiserwin');
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