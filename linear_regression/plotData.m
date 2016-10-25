function plotData(x, y)
  %   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
  %   population and profit.
  figure; % open a new figure window
  plot(x, y, 'rx');
  xlabel('population');
  ylabel('revenue');
endfunction
