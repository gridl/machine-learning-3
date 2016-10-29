function p = predict(Theta1, Theta2, X)
  %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  %   trained weights of a neural network (Theta1, Theta2)

  m = size(X, 1);
  num_labels = size(Theta2, 1);
  X = [ones(m,1), X];
  p = zeros(m, 1);
  a = sigmoid(Theta1*X');
  a = [ones(m, 1)'; a];
  z = sigmoid((Theta2*a)');
  [x, p] = max(z, [], 2);

endfunction
