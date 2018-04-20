% mu-law, frquency shifting, and plot functions modified/used 
% from sources found at http://eeweb.poly.edu/~yao/EE3414/
% Yao Wang, Polytechnic University, 2/11/2004

% perfrom nonlinear compression for mu-law
function [y_nonlinear] = mu_trasform(x, mu, Q)
    y_nonlinear = sign(x).*log(1+mu.*abs(x))./log(1+mu);
    %y_nonlinear = sign(x).*log10(1+abs(x)*(mu/xmax))/log10(1+mu);
end
