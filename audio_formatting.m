%function [song, fx, x_down, y_expanded, y_fx, z] = audio_formatting(bits)
function [song, fx, x_down, y_expanded, y_fx, z] = audio_formatting(bits)
%if nargin == 1
song_location = '/home/sable/HeaviNet/data/songs/ghost.mp3';

[song, fx, x_down, y_expanded, y_fx, z] = create_level(bits, song_location);

N = 2^(bits);
mu = N-1;
xmax = 1;
xmin = -1;
Q=(xmax-xmin)/N;

y_nonlinear = mu_trasform(x_down, mu, Q);
y_digital = analog2digital(y_nonlinear, Q);
y_analog = digital2analog(y_digital, Q);
y = mu_inverse(y_analog, mu, Q);

    D=x_down-y;
    MSE=mean(D.^2);
    fprintf('New MU error between original and quantized = %g\n',MSE )

end

function [song, fx, x_down, y_expanded, y_fx, z] = create_level(bits, song_location)
    data_location = '/home/sable/HeaviNet/data/input.mat';
    
    [song, fx] = audioread(song_location);
    song_info = audioinfo(song_location);
    if song_info.NumChannels == 2
        song = (song(:,1) + song(:,2))/2;
    end
    clipped_length =floor(length(song)/2^bits)*2^bits;
    song = song(1:clipped_length);

    N = 2^(bits);
    mu = N-1;
    fprintf('Bits:%d N:%d mu%d\n', bits,N,mu);

    
    y_fx = fx / 2^(9-bits);
    x_down = down_sample(song, bits);
    x_up = up_sample(x_down, bits);
    D=x_up-song;
    MSE=mean(D.^2);
    fprintf('Error between original and downsampled = %g\n',MSE )
    

    [y_expanded, y_compressed, Q] = mulaw(x_down, N, mu);
    
    z = up_sample(y_expanded, bits);
    
    D=z-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
    plot_q(x_down, y_expanded);
    %data = audioread(song_location, 'native');
    %save(data_location,'data');
end

% Modified from source Yao Wang, Polytechnic University, 2/11/2004
function [xq, yq, Q] = mulaw(x, N,mu)
    %magmax=max(abs(x));
    %xmin=-magmax, xmax=magmax;
    
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;


    %apply mu-law transform to original sample
    y=xmax*log10(1+abs(x)*(mu/xmax))/log10(1+mu);

    %apply uniform quantization on the absolute value each sample
    yq=floor((y-xmin)/Q)*Q+Q/2+xmin;
    %apply inverse mu-law transform to the quantized sequence
    %also use the original sign
    xq=(xmax/mu)*(10.^((log10(1+mu)/xmax)*yq)-1).*sign(x);


    %compare sound quality
    %audiowrite(outname, xq,fs);

    % Calculate the MSE
    D=x-xq;
    MSE=mean(D.^2);
    fprintf('Error between original and quantized = %g\n',MSE )
end

function [y_nonlinear] = mu_trasform(x, mu, Q)
    y_nonlinear = sign(x).*log(1+mu.*abs(x))./log(1+mu);
    %y_nonlinear = sign(x).*log10(1+abs(x)*(mu/xmax))/log10(1+mu);
end

function [y] = mu_inverse(y_nonlinear, mu, Q)
    xmax = 1;
    xmin = -1;
    y_quantized_nonlinear = floor((y_nonlinear-xmin)/Q)*Q+Q/2+xmin;
    y = sign(y_quantized_nonlinear).*(1/mu).*((1+mu).^(abs(y_quantized_nonlinear))-1);
    %y = (xmax/mu)*(10.^((log10(1+mu)/xmax)*y_nonlinear)-1).*sign(y_nonlinear);
end

function [y_digital] = analog2digital(y_nonlinear, Q)
    analog = y_nonlinear + 1;
    y_digital =floor(analog/Q);
end

function [y_nonlinear] = digital2analog(y_digital, Q)
    analog = y_digital*Q;
    y_nonlinear = analog -1;
end

function [z] = up_sample(y, bits)
z = y;
    for i = 0: (8-bits)
        z = interp(z,2, 8);
    end
end

function [y] = down_sample(x, bits)
y = x;
for i = 0: (8-bits)
    y = decimate(y,2,'fir');
end

%D=x-z;
%MSE=mean(D.^2);
%fprintf('\n Error between original and interpolated = %g\n\n',MSE )
end

function [] = plot_q(x, xq)
    %plot waveforms over the entire period
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
