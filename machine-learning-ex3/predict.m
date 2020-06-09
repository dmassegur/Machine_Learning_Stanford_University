function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(size(X, 1),1), X];

theta1x = X * Theta1';  % or theta1a1  or z2
a2 = sigmoid(theta1x);  % a2 = 1 ./ ( 1 + exp(-theta1x) );

a2 = [ones(size(a2, 1),1), a2];
theta2x = a2 * Theta2';  % or z3
prob = sigmoid(theta2x);  % prob = 1 ./ ( 1 + exp(-theta2x) );  % or a3

[i, y_pred] = max(prob,[],2);
p = y_pred;







% =========================================================================


end
