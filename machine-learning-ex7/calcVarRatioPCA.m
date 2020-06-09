function [VarRatio, VarRatioError] = calcVarRatioPCA(S, K, n)
% calcVarRatio computes the ratio of variances between K first eigenvalues and all n eigenvalues for PCA. 
%
% In particular:
%    VarRatio = sum of S from 1 to k / sum of S from 1 to n 
%    
% This is used to decide the minimum K for whiche a target variance is retained so the compressed data is good to use for training.

if n == 0
  n = size(S,2);  
endif


% VarRatio =  sum( sum( S(1:K,1:K) ) ) / sum( sum( S(1:n,1:n) ) );
VarRatio =  sum( diag( S(1:K,1:K) ) ) / sum( diag( S(1:n,1:n) ) );

VarRatioError = 1 - VarRatio;

  
endfunction
