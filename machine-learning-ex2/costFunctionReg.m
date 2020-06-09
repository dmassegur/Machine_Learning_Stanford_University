function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


##m = size(X,1);
##z = X*theta;
##htheta = sigmoid(z);
##log1 = log(htheta);
##log2 = log(1-htheta);
##
##reg = lambda*0.5/m * theta([1:end])'*theta([1:end]);
##
###J = -1/m * ( sum( y.*log1 + (1-y).*log2 ) ) + reg;
##J = -1/m * ( y'*log1 + (1-y)'*log2 ) + reg;
##
##
##reggrad = lambda/m * theta;
##reggrad[0] = 0;
##
##grad = 1/m * (htheta - y)'*X + reggrad;

# We call first the cost funciton calculation without the regularization and then we add the regularisation to it:
[J, grad] = costFunction(theta, X, y);

reg = lambda*0.5/m * theta([2:end])'*theta([2:end]);
J = J + reg;

reggrad = lambda/m * theta;
reggrad(1) = 0;
grad = grad + reggrad;




% =============================================================

end
