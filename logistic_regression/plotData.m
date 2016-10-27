function plotData(X, y)
  %PLOTDATA Plots the data points X and y into a new figure 
  %   PLOTDATA(x,y) plots the data points with + for the positive examples
  %   and o for the negative examples. X is assumed to be a Mx2 matrix.

  figure; hold on;
  negative = find(y == 0); positive = find(y == 1);
  plot(X(negative, 1), X(negative, 2), 'yo', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  plot(X(positive, 1), X(positive, 2), 'b+', 'LineWidth', 2, 'MarkerSize', 7);
  xlabel('x1')
  ylabel('y1')
  legend('Not admitted', 'Admitted')
  hold off;

endfunction
