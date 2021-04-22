
function [er,w,S1,V] = Effective_rank(A,r)
% (A,r) computing effective rank of A, with total proportion r (default 0.99)
[V,S] = eig(A);
S1 = diag(S);

if (nargin == 1)
    r = 0.99;
end;

Sum = sum(diag(S));
s = 0;
for i = 1:min(size(S))
    s = s + S(i,i);
    if (s/Sum >=r);
        er = i;
        break; 
    end;
end;

w = s/Sum;