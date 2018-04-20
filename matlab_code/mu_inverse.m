% perfrom nonlinear expansion for mu-law
function [y] = mu_inverse(y_nonlinear, mu, Q)
    xmax = 1;
    xmin = -1;
    y_quantized_nonlinear = floor((y_nonlinear-xmin)/Q)*Q+Q/2+xmin;
    y = sign(y_quantized_nonlinear).*(1/mu).*((1+mu).^(abs(y_quantized_nonlinear))-1);
    %y = (xmax/mu)*(10.^((log10(1+mu)/xmax)*y_nonlinear)-1).*sign(y_nonlinear);
end
