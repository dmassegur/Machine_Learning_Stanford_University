function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% J computation:

% X = [ones(m,1) , X];  % adding the bias terms

% Putting the Thetas in a big 3D matrix for vectorization:
Theta_dim (1,:) = size(Theta1);  % weights layer1 dimensions
Theta_dim (2,:) = size(Theta2);  % weights layer2 dimensions
ThetaMatrix = zeros( size(Theta_dim,1) , max(Theta_dim(:,1)), max(Theta_dim(:,2)) );
ThetaMatrix(1 , 1:Theta_dim(1,1) , 1:Theta_dim(1,2)) = Theta1;
ThetaMatrix(2 , 1:Theta_dim(2,1) , 1:Theta_dim(2,2)) = Theta2;
% size(ThetaMatrix);

% Setting y in vectors of length 10 with only the y index set to 1:
Ybin = zeros(m,max(y));
for i = 1 : m
  Ybin(i,y(i)) = 1;
end

##% Fwd Propagation: Computing h_theta non vectorised. This is actually Step 1 of the Back propagation for all the tests:
##z2 = X * Theta1';  % or theta1*a1  or theta1*x1
##a2 = [ones(m,1) , sigmoid(z2)];  % computing g(z) and adding the bias term
##
##z3 = a2 * Theta2';  % or theta2*a2
##htheta = sigmoid(z3);

% Fwd Propagation: Computing h_theta vectorised. This is actually Step 1 of the Back propagation for all the tests::
akplus1 = X;
Zk = zeros(size(ThetaMatrix,1) , size(X,1) , max([size(ThetaMatrix,2),size(ThetaMatrix,3)]) );
Ak = zeros(size(ThetaMatrix,1) , size(X,1) , max([size(ThetaMatrix,2),size(ThetaMatrix,3)]) );
% size(Zk)
% size(Ak)
for k = 1 : size(ThetaMatrix,1)  % loop through layers

  % fprintf('Computing Fwd propagation for layer %d.\n', k);
  ak = [ones(size(akplus1,1),1) , akplus1];
  % size(ak);
  % size(ThetaMatrix(k , 1:Theta_dim(k,1) , 1:Theta_dim(k,2)))
  Theta(:,:) = ThetaMatrix(k , 1:Theta_dim(k,1) , 1:Theta_dim(k,2));
  zkplus1 = ak * Theta';  % zk+1 = theta * ak
  akplus1 = sigmoid(zkplus1);  % computing g(z) and adding the bias term
  clear Theta;
  
  Zk( k , 1:size(zkplus1,1) , 2:size(zkplus1,2)+1 ) = zkplus1;
  Ak( k , 1:size(ak,1)      , 1:size(ak,2)        ) = ak;

endfor

hTheta = akplus1;

term1 = Ybin .* log(hTheta);
term2 = (1 - Ybin) .* log(1-hTheta);

J = -1/m * sum(sum(term1+term2));


% Regularization of J:

% we can't use nn_params because these include theta0 too! and regularzation exclude the bias thetas!!!
##Theta1nobias = Theta1(:,2:end);
##Theta2nobias = Theta2(:,2:end);
##unrolltheta = [Theta1nobias(:) ; Theta2nobias(:)];

% Unrolling Thetas for regularization, excluding the bias weights!!!
unrolltheta = [];
for k = 1 : size(ThetaMatrix,1)

  % fprintf('Unrolling layer %d weights, excluding the bias terms.\n', k);
  theta(:,:) = ThetaMatrix(k , 1:Theta_dim(k,1) , 2:Theta_dim(k,2));
  unrolltheta = [unrolltheta ; theta(:)];
  % size(unrolltheta);
  clear theta
  
endfor

regterm = 0.5 * lambda / m * ( unrolltheta' * unrolltheta );
J += regterm;


% Backpropagation for Gradient computation:

##%%% Non-vectorized backpropagation
##for i = 1 : size(X,1)   % loop through training test smaples
##
##  % Step 1:
##% No need to compute Step 1 because it's already available above for all the training tests.  
##  a1(1,:) = [1 , X(i,:)];  % we add the bias term!
##  z2 = a1 * Theta1';
##  a2 = [1 , sigmoid(z2)];
##  z3 = a2 * Theta2';  % not used
##  a3 = sigmoid(z3);  % not used
##  
##  % Step 2:
##  delta3(1,:) = hTheta(i,:) - Ybin(i,:);  % or a3 - Ybin(i,:)
##  
##  % Step 3:
##  theta(:,:) = ThetaMatrix( 2 , 1:Theta_dim(2,1) , 2:Theta_dim(2,2) );
##  delta2 = delta3 * theta .* sigmoidGradient(z2);
##  
##  % Step 4:  Delta computation
##  Theta2_grad += delta3' * a2;  
##
##  %delta2 = delta2(1,2:end);  % not needed
##  Theta1_grad += delta2' * a1;    
##   
##endfor
##
##  % Step 5
##Theta1_grad = Theta1_grad / m;
##Theta2_grad = Theta2_grad / m;
##
##
##% Grad J Regularization:
##Grad_reg = lambda / m * ThetaMatrix;
##
##Grad_reg1 = zeros(Theta_dim(1,1) , Theta_dim(1,2));
##gr(:,:) = Grad_reg(1 , 1:Theta_dim(1,1) , 2:Theta_dim(1,2));
##Grad_reg1(:,2:end) += gr;
##Theta1_grad += Grad_reg1;
##clear gr;
##
##Grad_reg2 = zeros(Theta_dim(2,1) , Theta_dim(2,2));
##gr(:,:) = Grad_reg(2 , 1:Theta_dim(2,1) , 2:Theta_dim(2,2));
##Grad_reg2(:,2:end) += gr;
##Theta2_grad += Grad_reg2;
##clear gr;


%%% Vectorised backpropagation (for all training examples):

  % Step 1:
  % done above during computation of J

  % Step 2:  % last layer error deltak  (a3 - yk)
%Deltak = zeros(size(ThetaMatrix,1) , size(ThetaMatrix,2) , size(ThetaMatrix,3) );
%Deltak(end,size(hTheta,1),size(hTheta,2)) = hTheta - Ybin;  % a3 - yk
Deltak = hTheta - Ybin; % a3 - yk

ThetaGradkij = zeros(size(ThetaMatrix));
for k = size(ThetaMatrix,1):-1:1   % backward loop through the layers
  
  % Step 3: loop backwards through the other layers
  Theta(:,:) = ThetaMatrix( k , 1:Theta_dim(k,1) , 2:Theta_dim(k,2) );  % for computation of delta we can exclude the theta0
  Deltakplus1 = Deltak; %(k,1:size(ThetaMatrix,1),1:size(ThetaMatrix,2));
  if k>1 
    Z(:,:) = Zk(k-1,1:size(X,1),2:Theta_dim(k,2));  % Zk does not contain the bias term
    Deltak = Deltakplus1 * Theta .* sigmoidGradient(Z);
  endif
  
  % Step 4:  Delta computation
  A(:,:) = Ak(k,1:size(hTheta,1),1:Theta_dim(k,2));
  
  DeltaA = zeros(size(ThetaMatrix));
  for i = 1:size(X,1)  % or m 
    deltaa = Deltakplus1(i,:)' * A(i,:); 
    DeltaA(k,1:size(deltaa,1),1:size(deltaa,2)) = deltaa;
    ThetaGradkij += DeltaA;
  endfor
    
  clear Theta Deltakplus1 Z A;
  
endfor

  % Step 5:  sum for all the training samples
ThetaGradkij /= m;


% Grad J Regularization vectorised:
Grad_reg = lambda / m * ThetaMatrix;
Grad_reg(:,:,1) = 0;
ThetaGradkij += Grad_reg;


Theta1_grad = ThetaGradkij(1,1:Theta_dim(1,1),1:Theta_dim(1,2));
Theta2_grad = ThetaGradkij(2,1:Theta_dim(2,1),1:Theta_dim(2,2));  




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
