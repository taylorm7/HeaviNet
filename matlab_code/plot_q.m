%plot waveforms, comparing quantized to original
function [] = plot_q(x, xq)
    
    t=1:length(x);
    figure; plot(t,x,'r:');
    hold on; plot(t,xq,'b-');
    axis tight; grid on;
    legend('original','quantized')
    %plot waveform over a selected period
    t=2000:2200;
    figure; plot(t,x(2000:2200),'r:');
    hold on; plot(t,xq(2000:2200),'b-');
    axis tight; grid on;
end
