

clear all;
t_init = 0;
tic;
%% A pde solver (DiffCoef with randomness)
% Domain : unit square 'rect'
% PDE : -Div( DiffCoef * Grad(u)) = f (**)

% Boundary condition : Dirchelet zero
g_D = @(x,y)  0;
g_N = @(x,y)  0;
Kappa = @(x,y) 1e10;

% FEM mesh
N = 2^9;
h = 1/N;
[X,Y] = ndgrid((0:N)/N,(0:N)/N);
p = [X(:) Y(:)];
t = delaunay(p);
e = freeBoundary(triangulation(t,p));

% Robin matrix, Load vector, Robin vector (because mesh, Bd Cond, RHS are fixed)
R1 = RobinMat2D_sparse(p,e,Kappa);
H1 = StiffMat2D_sparse(@(x,y)1,p,t);
M1 = MassMat2D_sparse(p,t);
r1 = RobinVec2D_sparse(p,e,Kappa,g_D,g_N);


% subdomains, 左下横坐标，左下纵坐标，宽度，高度，local区域的指标集
Do_2 = [1/4  1/16    1/2  1/4];
Do_1 = round([1/4  11/16  1/2  1/4]*N+[1 1 0 0]);
[loc_idxX,loc_idxY] = ndgrid(Do_1(1):Do_1(1)+Do_1(3),Do_1(2):Do_1(2)+Do_1(4));
loc_idx = (reshape(loc_idxY,[],1)-1) * (N+1) + reshape(loc_idxX,[],1);

% % setting of RHS function
% RightF = @(x,y)  ( cos(pi*2*x) * sin(pi*2*y) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
% b1 = LoadVec2D_sparse(p,t,RightF); 

% setting of Diffusion Coefficient
angle_coef = pi*(1:18)'/18;
epsilon = 1./(11:2:45)';
     
t_init = toc;
%% 
training_size = 5000;
K2 = round(training_size * 5);
Xonline = zeros(20,K2);
parfor k = 1:K2
    xi = rand(18,1)*2-1;
    theta = [rand(1,1)* Do_2(3) + Do_2(1) ; rand(1,1) * Do_2(4) + Do_2(2)];
    Xonline(:,k) = [xi;theta];
end;
    


save('revision1_Ex4_local_data','Xonline','K2','-v7.3');


