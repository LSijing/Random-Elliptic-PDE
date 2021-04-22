
%% -------boundary condition vector-------%
%   boundary condition : -n.*(a * Grad(u)) = kappa(u-g_D) - g_N
%   discrete form (A+R)u = b+r
%   r_ij = integral_partial{Omega}{(kappa * g_D + g_N) * phi_i}
%   computing Robin BC vector

function [r] = RobinVec2D_sparse(p,e,kappa,gD,gN)
np = size(p,1);
ne = size(e,1);


x1_bd = p(e(:,1),1);
x2_bd = p(e(:,2),1);
y1_bd = p(e(:,1),2);
y2_bd = p(e(:,2),2);
len = sqrt((x1_bd-x2_bd).^2+(y1_bd-y2_bd).^2);
xc_bd = (x1_bd + x2_bd)/2;
yc_bd = (y1_bd + y2_bd)/2;
kbar = arrayfun(kappa,xc_bd,yc_bd);


Isparse_Robinvec = reshape(e',[],1);
Jsparse_Robinvec = ones(ne*2,1);
temp = kbar .* arrayfun(gD,xc_bd,yc_bd) + arrayfun(gN,xc_bd,yc_bd);
rK = temp .* len /2;
rsparse_Robinvec = reshape(repmat(rK,1,2)',[],1);
r = sparse(Isparse_Robinvec,Jsparse_Robinvec,rsparse_Robinvec,np,1);
