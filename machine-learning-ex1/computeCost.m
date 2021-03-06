function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

##for i = 1:size(X,2)
##  avg_X1 = mean(X(:,i));
##  range_x1 = max(X(:,i)) - min(X(:,i));
##endfor
##normalized_X = ( X(:,i) - avg_X1 ) ./ range_x1;  %normalized and scaled 


error_sq = ((theta'*X')' - y) .^2;
%error_sq = (X*theta - y) .^2;
J = 0.5/m*sum(error_sq,1);




% =========================================================================

end
