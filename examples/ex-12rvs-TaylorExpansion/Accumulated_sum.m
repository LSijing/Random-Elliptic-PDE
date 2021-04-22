
function [w,s,V] = Accumulated_sum(A,R)
% (A,r) computing accumulated sum of weights from 1 to R. 
[V,S] = eig(A);

s = diag(S);
w = sum(s(1:R))/sum(s);