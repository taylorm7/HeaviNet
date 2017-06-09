% '/home/sable/Music/beet.wav'
% '/home/sable/Music/clams_u.wav'
% '/home/sable/AudioFiltering/Waves/Sine.wav';


song_location = '/home/sable/Music/clams_u.wav';
data_location = '/home/sable/HeaviNet/data/input.mat';

[song, song_rate] = audioread(song_location);
song_info = audioinfo(song_location);
data = audioread(song_location, 'native');

song_clip = song(song_rate*1:song_rate*5,:);
data_clip = data(song_rate*1:song_rate*5,:);

save(data_location,'data');