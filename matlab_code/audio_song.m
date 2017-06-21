% Format audio for level structure for use in HeaviNet Neural Network
% for more information visit https://github.com/taylorm7/HeaviNet

function [song, fx, x_down6, x_fx6, y6, level_6, ytrue_6, y_final7, z6] = audio_song(bits, song_location, data_location)
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
