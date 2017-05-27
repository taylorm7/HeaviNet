% '/home/sable/Music/beet.wav'
% '/home/sable/Music/clams.mp3'
% '/home/sable/AudioFiltering/Waves/Sine.wav';


song_location = '/home/sable/AudioFiltering/Waves/Sine_u.wav';
data_location = '/home/sable/AudioFiltering/Testing/test.mat';

[song, song_rate] = audioread(song_location);
song_info = audioinfo(song_location);
data = audioread(song_location, 'native');

song_clip = song(song_rate*10:song_rate*15,:);

save(data_location,'data');