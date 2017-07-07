% convert compressed analog signal to uniform integer values
function [y_digital] = analog_to_digital(y_nonlinear, Q)
    analog = y_nonlinear + 1;
    y_digital = floor(analog/Q);
    % exclude any digital values out of range (-1,1)
    y_digital(y_digital >= 2/Q) = (2-Q)/Q;
    y_digital(y_digital < 0) = 0;
    y_digital = int32(y_digital);
end
