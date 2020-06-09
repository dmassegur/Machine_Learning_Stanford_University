function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Note: X must already contain the bias term!!!

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Non-Regularised Cost Function:
J = 0.5 / m * ( X * theta - y )' * ( X * theta - y );

% Non-Regularised Gradients:
grad = 1 / m * X' * ( X * theta - y ); 


% Regularised Cost Function:
reg = 0.5 * lambda / m * theta(2:end)' * theta(2:end);   % bias term excluded
J += reg;
clear reg;

% Regularised Gradients:
reg = lambda / m * theta;
reg(1) = 0.0;   % bias term excluded
grad += reg; 










% =========================================================================

grad = grad(:);

end
