function [song_clipped, x_down, fx, x_expanded, y_compressed, z] = audio_formatting(bits, song_location, data_location)
    if 1 %nargin == 2 
        data_location = '/home/sable/HeaviNet/data/input.mat';
        song_location = '/home/sable/HeaviNet/data/songs/ghost.mp3';
    %elseif nargin == 2
        %data_location = '/home/sable/HeaviNet/data/input.mat';
    end

    [song, song_rate] = audioread(song_location);
    song_info = audioinfo(song_location);
    if song_info.NumChannels == 2
        song = (song(:,1) + song(:,2))/2;
    end
    
    clipped_length =floor(length(song)/2^bits)*2^bits;
    song_clipped = song(1:clipped_length);

    N = 2^(bits);
    mu = N-1;
    
    disp('Bits,N,mu,Q,');
    disp([bits,N,mu]);
    
    fx = song_info.SampleRate / 2^(9-bits);
    %x = resample(song, fx, song_rate);
    x_down = down_sample(song_clipped, bits);
    x_up = up_sample(x_down, bits);
    D=x_up-song_clipped;
    MSE=mean(D.^2);
    fprintf('\n Error between original and downsampled = %g\n\n',MSE )
    

    [x_expanded, y_compressed, Q] = mulaw(x_down, N, mu, song_clipped);
    
    z = up_sample(x_expanded, bits);
    
    D=z-song_clipped;
    MSE=mean(D.^2);
    fprintf('\n Error between original and final = %g\n\n',MSE )
    
    %plot_q(x, x_expanded);
    %data = audioread(song_location, 'native');
    %save(data_location,'data');
end

% Modified from source Yao Wang, Polytechnic University, 2/11/2004
function [xq, yq, Q] = mulaw(x, N,mu, s)
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
    fprintf('\n Error between original and quantized = %g\n\n',MSE )
end

function [z] = up_sample(y, bits)
z = y;
    for i = 0: (9-bits)
        z = interp(z,2);
    end
end

function [y] = down_sample(x, bits)
y = x;
for i = 0: (9-bits)
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
