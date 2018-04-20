% convert uniform integer values into analog signal
function [y_nonlinear] = digital_to_analog(y_digital, Q)
    analog = double(y_digital)*Q;
    y_nonlinear = analog -1;
end

