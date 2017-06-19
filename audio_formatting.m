% Format audio for level structure for use in HeaviNet Neural Network
% for more information visit https://github.com/taylorm7/HeaviNet

function [song, fx, x_down6, x_fx6, y6, level_6, ytrue_6, y_final7, z6] = audio_formatting(bits, song_location, data_location)
%if nargin == 1
%song_location = '/home/sable/HeaviNet/data/songs/voice.wav';pieces_pittoresques
%song_name = 'voice.wav';
%song_location = '/home/sable/HeaviNet/data/songs/' + song_name;
disp(song_location);
disp(data_location);
% read from song location, make audio mono, and clip to valid length for
% downsampling
[song, fx] = audioread(song_location);
song_info = audioinfo(song_location);
if song_info.NumChannels == 2
    song = (song(:,1) + song(:,2))/2;
end
downsample_rate = 0;
clipped_length =floor(length(song)/2^(bits+downsample_rate))*2^(bits+downsample_rate);
song = song(1:clipped_length);
[song] = down_sample(song, downsample_rate);
fx = fx/2^downsample_rate;

%create level based on number of bits for sampling
[x_down0, x_fx0, y0, level_0, z0] = create_level(1, song, fx, bits);
[x_down1, x_fx1, y1, level_1, z1] = create_level(2, song, fx, bits);
[x_down2, x_fx2, y2, level_2, z2] = create_level(3, song, fx, bits);
[x_down3, x_fx3, y3, level_3, z3] = create_level(4, song, fx, bits);
[x_down4, x_fx4, y4, level_4, z4] = create_level(5, song, fx, bits);
[x_down5, x_fx5, y5, level_5, z5] = create_level(6, song, fx, bits);
[x_down6, x_fx6, y6, level_6, z6] = create_level(7, song, fx, bits);
%[x_down7, x_fx7, y7, level_7, z7] = create_level(8, song, fx, bits);
%[x_down8, x_fx8, y8, level_8, z8] = create_level(9, song, fx, bits);

[ytrue_0] = create_solution(1, song, fx, bits);
[ytrue_1] = create_solution(2, song, fx, bits);
[ytrue_2] = create_solution(3, song, fx, bits);
[ytrue_3] = create_solution(4, song, fx, bits);
[ytrue_4] = create_solution(5, song, fx, bits);
[ytrue_5] = create_solution(6, song, fx, bits);
[ytrue_6, y_final7] = create_solution(7, song, fx, bits);
%[ytrue_8,yt8] = create_solution(8, song, fx, bits);

save(data_location, 'song_location', 'level_0', 'ytrue_0', 'level_1', 'ytrue_1', 'level_2', 'ytrue_2', 'level_3', 'ytrue_3','level_4','ytrue_4','level_5', 'ytrue_5', 'level_6', 'ytrue_6' );

end

function [y_digital,y] = create_solution(level, song, fx, total_levels)

    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(level + 1);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    fx_factor = total_levels - level;
    x_fx = fx / 2^(fx_factor);
    
    fprintf('Solution Level:%d bits:%d fx:%g N:%d mu:%d Q:%g \n', level-1, level,x_fx, N,mu, Q);
    
    % downsample and upsample based on bit level 
    % (higher precision, lower downsampling)
    
    x_down = down_sample(song, fx_factor);
    x_up = up_sample(x_down, fx_factor);
    
    % compute error for downsampling and upsampling
    D=x_up-song;
    MSE=mean(D.^2);
    fprintf('Error between original and downsampled = %g\n',MSE )

    % perform mu-law transform and digitize compressed data
    y_nonlinear = mu_trasform(x_down, mu, Q);
    y_digital = analog2digital(y_nonlinear, Q);
    % compute analog to digital, and perform inverse mu-law transform
    y_analog = digital2analog(y_digital, Q);
    y = mu_inverse(y_analog, mu, Q);
    
    % compute error for mu-law transform
    D=x_down-y;
    MSE=mean(D.^2);
    fprintf('Error between original and quantized = %g\n',MSE )
    
    % upsample companded audio to origianl sampling frequency
    z = up_sample(y, fx_factor);
    
    % compute final error for companded and digitized data
    D=z-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
    %plot and write data to _.mat file
    %plot_q(x_down, y);

end

function [x_down, x_fx, y, y_digital, z] = create_level(level, song, fx, total_levels)
    
    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(level);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    fx_factor = total_levels - level;
    x_fx = fx / 2^(fx_factor);
    
    fprintf('Level:%d bits:%d x_fx:%g N:%d mu:%d Q:%g \n', level-1, level,x_fx, N,mu, Q);
    
    % downsample and upsample based on bit level 
    % (higher precision, lower downsampling)
    
    x_down = down_sample(song, fx_factor);
    x_up = up_sample(x_down, fx_factor);
    
    % compute error for downsampling and upsampling
    D=x_up-song;
    MSE=mean(D.^2);
    fprintf('Error between original and downsampled = %g\n',MSE )

    % perform mu-law transform and digitize compressed data
    y_nonlinear = mu_trasform(x_down, mu, Q);
    y_digital = analog2digital(y_nonlinear, Q);
    % compute analog to digital, and perform inverse mu-law transform
    y_analog = digital2analog(y_digital, Q);
    y = mu_inverse(y_analog, mu, Q);
    
    % compute error for mu-law transform
    D=x_down-y;
    MSE=mean(D.^2);
    fprintf('Error between original and quantized = %g\n',MSE )
    
    % upsample companded audio to origianl sampling frequency
    z = up_sample(y, fx_factor);
    
    % compute final error for companded and digitized data
    D=z-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
    %plot and write data to _.mat file
    %plot_q(x_down, y);

end

% mu-law, frquency shifting, and plot functions modified/used 
% from sources found at http://eeweb.poly.edu/~yao/EE3414/
% Yao Wang, Polytechnic University, 2/11/2004

% perfrom nonlinear compression for mu-law
function [y_nonlinear] = mu_trasform(x, mu, Q)
    y_nonlinear = sign(x).*log(1+mu.*abs(x))./log(1+mu);
    %y_nonlinear = sign(x).*log10(1+abs(x)*(mu/xmax))/log10(1+mu);
end

% perfrom nonlinear expansion for mu-law
function [y] = mu_inverse(y_nonlinear, mu, Q)
    xmax = 1;
    xmin = -1;
    y_quantized_nonlinear = floor((y_nonlinear-xmin)/Q)*Q+Q/2+xmin;
    y = sign(y_quantized_nonlinear).*(1/mu).*((1+mu).^(abs(y_quantized_nonlinear))-1);
    %y = (xmax/mu)*(10.^((log10(1+mu)/xmax)*y_nonlinear)-1).*sign(y_nonlinear);
end

% convert compressed analog signal to uniform integer values
function [y_digital] = analog2digital(y_nonlinear, Q)
    analog = y_nonlinear + 1;
    y_digital = floor(analog/Q);
    % exclude any digital values out of range (-1,1)
    y_digital(y_digital >= 2/Q) = (2-Q)/Q;
    y_digital(y_digital < 0) = 0;
    y_digital = int32(y_digital);
end

% convert uniform integer values into analog signal
function [y_nonlinear] = digital2analog(y_digital, Q)
    analog = double(y_digital)*Q;
    y_nonlinear = analog -1;
end

% upsample with 'fir' filter with window size 8
% sampling frequency is equal to  origianl_fx/2^(9-bits)
% inversevly proportionate to precision of data
function [z] = up_sample(y, factor)
z = y;
    for i = 1: factor
        z = interp(z,2, 8);
    end
end

% downsample with default 'fir' filter of n=30
% sampling frequency is equal to  origianl_fx/2^(9-bits)
% inversevly proportionate to precision of data
function [y] = down_sample(x, factor)
y = x;
    for i = 1:factor
        y = decimate(y,2,'fir');
    end
end

%plot waveforms, comparing quantized to original
function [] = plot_q(x, xq)
    
    t=1:length(x);
    figure; plot(t,x,'r:');
    hold on; plot(t,xq,'b-');
    axis tight; grid on;
    legend('original','quantized')
    %plot waveform over a selected period
    t=2000:2200;
    figure; plot(t,x(2000:2200),'r:');
    hold on; plot(t,xq(2000:2200),'b-');
    axis tight; grid on;
end
