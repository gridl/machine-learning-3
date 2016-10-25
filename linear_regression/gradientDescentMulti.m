function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
      S = (X*theta - y).*X;
      S = S*(alpha/m);
      theta = theta - sum(S)';
      % Save the cost J in every iteration    
      J_history(iter) = computeCostMulti(X, y, theta);
  endfor
endfunction
