function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% training set size:
m = size(X,1);

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

## Computing the centroid assignment for each training example via a loop:
for i =  1 : m
  dist_bsf = 100000;
  for j =  1 : K
      
      % computing the norm of the distance between example x_i and centroid mu_j
      dist = X(i,:) - centroids(j,:);
      dist = norm(dist);
      
      % sorting whether exampli i is closer to current centroid j or not. IF yes, update j index for example i:
      if dist < dist_bsf
        dist_bsf = dist;
        idx(i,1) = j;
      endif
  endfor
endfor

## Attempt to vectorise it:
##Xbig = X;
##for j = 2 : K
##  Xbig = [Xbig, X];
##endfor
##
##centr_col = centroids'(:)';
##centr_big = centr_col;
##for i = 2 : m
##  centr_big = [centr_big, centr_col];
##endfor
##
##size(Xbig);
##size(centr_big)
##
##delx = Xbig - centr_big;







% =============================================================

end

