
%% --------assemble global boundary condition matrix R--------%
%   boundary condition : -n.*(a * Grad(u)) = kappa(u-g_D) - g_N
%   discrete form (A+R)u = b+r
%   R_ij = integral_partial{Omega}{kappa * phi_i * phi_j}
%   computing Robin BC matrix

function [R] = RobinMat2D_sparse(p,e,Kappa)

np = size(p,1);
ne = size(e,1);

Isparse_Robin = reshape(repmat(e,1,2)',[],1);
Jsparse_Robin = reshape(repmat(reshape(e',[],1),1,2)',[],1);
x1_bd = p(e(:,1),1);
x2_bd = p(e(:,2),1);
y1_bd = p(e(:,1),2);
y2_bd = p(e(:,2),2);
len = sqrt((x1_bd-x2_bd).^2+(y1_bd-y2_bd).^2);
xc_bd = (x1_bd + x2_bd)/2;
yc_bd = (y1_bd + y2_bd)/2;
kbar = arrayfun(Kappa,xc_bd,yc_bd);
RK = repmat([2 1 1 2],ne,1).*repmat(len,1,4).*repmat(kbar,1,4) /6;
Rsparse_Robin = reshape(RK',[],1);
R = sparse(Isparse_Robin,Jsparse_Robin,Rsparse_Robin,np,np);

