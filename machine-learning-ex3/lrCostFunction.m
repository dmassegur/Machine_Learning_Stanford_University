function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% non-regularised J:
thetax = X * theta;
sigm = sigmoid(thetax);  % sigm = 1 ./ ( 1 + exp(-thetax));
Jterm1 = y' * log(sigm);
Jterm2 = (1-y)' * log(1-sigm);
J = -1/m * (Jterm1 + Jterm2);

% regularised J:
J += 0.5 * lambda / m * theta(2:end)' * theta(2:end);


% non-regularised gradient:
sigm_minus_y = sigm - y;
grad = (1/m * sigm_minus_y' * X)'; % or 1/m * X' * sigm_minus_y;

% regularised gradient:
lamterm = lambda / m * theta;
lamterm(1) = 0;
% alternatively: lamterm(2:length(theta)) = lambda / m .* theta(2:length(theta));
% or: lamterm(2:end) = lambda / m .* theta(2:end);
grad += lamterm;









% =============================================================

grad = grad(:);

end
