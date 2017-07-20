function [finished_song, finished_fx , input_digital] = audio_finish(read_location, write_location, data_location, level, downsample_rate)
    disp(read_location);
    disp(write_location);
    
    level = level + 1
    
    [passband_fx, original_fx] = get_fx(data_location, level);
    
    N = 2^(8);
    mu = N-1;
    xmax = 1;
    xmin = -1;
    Q=(xmax-xmin)/N;
    
    input_digital = transpose(importdata(read_location));
    input_analog = digital_to_analog(input_digital, Q);
    finished_song = mu_inverse(input_analog, mu, Q);
    if downsample_rate > 1
        finished_fx = round(original_fx/downsample_rate);
    else
        finished_fx = original_fx;
    end
    
    disp('Song saved at');
    audiowrite(write_location, finished_song, finished_fx);
    final_info = audioinfo(write_location)
    
    
end
