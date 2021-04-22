
%% ------assemble right hand side-------%
%  equation : - Div( a * Gradient (u) ) = f
%  variational form : integral{a * Grad(phi_i) * sum{Grad(phi_j)*u_j} } = integral{f * phi_i}
%  vector 'LoadVec2D_sparse' here is right side 'int{f * phi_i}' in above

%%
function [b] = LoadVec2D_sparse(p,t,f)

np = size(p,1);
nt = size(t,1);


x1_elem = p(t(:,1),1);
x2_elem = p(t(:,2),1);
x3_elem = p(t(:,3),1);
y1_elem = p(t(:,1),2);
y2_elem = p(t(:,2),2);
y3_elem = p(t(:,3),2);
area = polyarea([x1_elem';x2_elem';x3_elem'],[y1_elem';y2_elem';y3_elem'])';
xc_elem = (x1_elem + x2_elem + x3_elem) /3;
yc_elem = (y1_elem + y2_elem + y3_elem) /3;


Isparse_loadv = reshape(t',[],1);
Jsparse_loadv = ones(nt*3,1);
% (center on three edges)
x_ec = [(x1_elem+x2_elem)/2  (x2_elem+x3_elem)/2  (x3_elem+x1_elem)/2];
y_ec = [(y1_elem+y2_elem)/2  (y2_elem+y3_elem)/2  (y3_elem+y1_elem)/2];
fbar = ( arrayfun(f,x_ec(:,1),y_ec(:,1)) + arrayfun(f,x_ec(:,2),y_ec(:,2)) + arrayfun(f,x_ec(:,3),y_ec(:,3)) )/3;
%fbar = arrayfun(RightF,xc_elem,yc_elem);
% 如何近似右端项？f(xc,yc)太粗糙。(f(x_ec1,y_ec1)+f(x_ec2,y_ec2)+f(x_ec3,y_ec3))使fbar更精确了，但依然是对三个顶点的贡献相同，也不符合实际。
bK = fbar .* area /3;
bsparse_loadv = reshape(repmat(bK,1,3)',[],1);
b = sparse(Isparse_loadv,Jsparse_loadv,bsparse_loadv,np,1);

