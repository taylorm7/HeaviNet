function [] = audio_formatting(bits)
e1 = create_level(bits, '/home/sable/HeaviNet/data/songs/yellow.mp3');
e2 = create_level(bits, '/home/sable/HeaviNet/data/songs/mind.mp3');
e3 = create_level(bits, '/home/sable/HeaviNet/data/songs/ghost.mp3');
e4 = create_level(bits, '/home/sable/HeaviNet/data/songs/clams.mp3');
e5 = create_level(bits, '/home/sable/HeaviNet/data/songs/manila.mp3');
error = e1 + e2+e3+e4+e5;
fprintf("error:%g\n", error);
end

function [MSE, song, fx, x_down, y_expanded, y_fx, z] = create_level(bits, song_location, data_location)
    if 1 %nargin == 2 
        data_location = '/home/sable/HeaviNet/data/input.mat';
        %song_location = '/home/sable/HeaviNet/data/songs/yellow.mp3';
    %elseif nargin == 2
        %data_location = '/home/sable/HeaviNet/data/input.mat';
    end

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
    

    [y_expanded, y_compressed, Q] = mulaw(x_down, N, mu, song);
    
    z = up_sample(y_expanded, bits);
    
    D=z-song;
    MSE=mean(D.^2);
    fprintf('Error between original and final = %g\n',MSE )
    
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
    fprintf('Error between original and quantized = %g\n',MSE )
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
