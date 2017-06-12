function [x, fx, x_expanded, y_compressed, Q] = audio_formatting(bits, song_location, data_location)
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
    
    fx = 8000;
    x = resample(song, fx, song_rate);
    
    N = 2^(bits);
    mu = N-1;
    [x_expanded, y_compressed, Q] = mulaw(x, fx, N, mu);
    
    data = audioread(song_location, 'native');
    save(data_location,'data');
end

% Modified from source Yao Wang, Polytechnic University, 2/11/2004
function [xq, yq, Q] = mulaw(x, fs, N,mu)
    %magmax=max(abs(x));
    %xmin=-magmax, xmax=magmax;
    
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    disp('xmin,xmax,N,mu,Q,');
    disp([xmin,xmax,N,mu,Q]);

    %apply mu-law transform to original sample
    y=xmax*log10(1+abs(x)*(mu/xmax))/log10(1+mu);

    %apply uniform quantization on the absolute value each sample
    yq=floor((y-xmin)/Q)*Q+Q/2+xmin;

    %apply inverse mu-law transform to the quantized sequence
    %also use the original sign
    xq=(xmax/mu)*(10.^((log10(1+mu)/xmax)*yq)-1).*sign(x);


    %compare sound quality
    %audiowrite(outname, xq,fs);

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

    % Calculate the MSE
    D=x-xq;
    MSE=mean(D.^2);
    fprintf('\n Error between original and quantized = %g\n\n',MSE )
end

