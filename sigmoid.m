function g = sigmoid(z)

% SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1 ./ (1 + exp(-z));

g(isnan(g),:) = -10^5;  % representing -inf

end
