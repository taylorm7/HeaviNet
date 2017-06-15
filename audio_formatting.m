% Format audio for level structure for use in HeaviNet Neural Network
% for more information visit https://github.com/taylorm7/HeaviNet

function [song, fx, x_down, x_fx, y, y_d, z] = audio_formatting(bits)
%if nargin == 1
song_location = '/home/sable/HeaviNet/data/songs/clams.mp3';
% read from song location, make audio mono, and clip to valid length for
% downsampling
[song, fx] = audioread(song_location);
song_info = audioinfo(song_location);
if song_info.NumChannels == 2
    song = (song(:,1) + song(:,2))/2;
end
clipped_length =floor(length(song)/2^bits)*2^bits;
song = song(1:clipped_length);

%create level based on number of bits for sampling
[x_down, x_fx, y, y_d, z] = create_level(bits, song, fx);

end

function [x_down, x_fx, y, y_digital, z] = create_level(bits, song, fx)
    data_location = '/home/sable/HeaviNet/data/input.mat';
    
    % set values for mu transform, assuming standard -1,1 audio
    N = 2^(bits);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    fprintf('Bits:%d N:%d mu:%d Q:%g \n', bits,N,mu, Q);
    
    % downsample and upsample based on bit level 
    % (higher precision, lower downsampling)
    x_fx = fx / 2^(9-bits);
    x_down = down_sample(song, bits);
    x_up = up_sample(x_down, bits);
    
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
    z = up_sample(y, bits);
    
    % compute final error for companded and digitized data
    D=z-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
    %plot and write data to _.mat file
    plot_q(x_down, y);
    %data = audioread(song_location, 'native');
    %save(data_location,'data');
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
function [z] = up_sample(y, bits)
z = y;
    for i = 0: (8-bits)
        z = interp(z,2, 8);
    end
end

% downsample with default 'fir' filter of n=30
% sampling frequency is equal to  origianl_fx/2^(9-bits)
% inversevly proportionate to precision of data
function [y] = down_sample(x, bits)
y = x;
    for i = 0: (8-bits)
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
