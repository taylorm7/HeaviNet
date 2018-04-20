% audio_read: read wav file at location according to downsample_rate
% inputs:
% song_location, string of audio file to read
% downsampling_rate, rate to downsample audio
% outputs:
% song, matrix of double values for the wav file in mono
% fx, sampling frequency of song
function [song, fx ] = audio_read(song_location, downsample_rate)
[song, fx] = audioread(song_location);
song_info = audioinfo(song_location);
if song_info.NumChannels == 2
    song = (song(:,1) + song(:,2))/2;
end

if downsample_rate > 1
    clipped_length =floor(length(song)/downsample_rate)*downsample_rate;
    song = song(1:clipped_length);
    song = decimate(song,downsample_rate,'fir');
    fx = fx/downsample_rate;
end

end