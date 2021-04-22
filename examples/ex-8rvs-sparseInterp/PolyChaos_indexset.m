
function [idx] = PolyChaos_indexset(d, p);
%% generate index set recursively
% d : number of dimensions
% p : total maximal order
% aim to generate \sum_1^d |idx_i| <= p, 
% total order (d+p)!/(d!*p!)
% each column is a set of index
%% 
% ending condition 1 of recursion
if (d==1) 
    idx = 0:p;
    return;
end;
% ending condition 1 of recursion
if (p==0)
    idx = zeros(d,1);
    return;
end;

% recursion
idx = [];
for k = 0:p
    idx2 = PolyChaos_indexset(d-1,p-k);
    n = size(idx2,2);
    idx = [idx,[ones(1,n)*k;idx2]];
end;

