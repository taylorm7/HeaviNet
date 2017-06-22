function [finished_song, finished_fx , input_digital] = audio_finish(read_location, write_location, original_location, level)
    disp(read_location);
    disp(write_location);
    
    original_info = audioinfo(original_location)
    original_fx = original_info.SampleRate 
    
    N = 2^(level);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    input_digital = transpose(importdata(read_location));
    input_analog = digital_to_analog(input_digital, Q);
    finished_song = mu_inverse(input_analog, mu, Q);
    finished_fx = original_fx / 1;
    audiowrite(write_location, finished_song, finished_fx);
end
