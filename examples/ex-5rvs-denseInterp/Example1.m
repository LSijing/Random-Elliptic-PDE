clear all;

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

% setting of Diffusion Coefficient
epsilon = [1/47 1/29 1/53 1/37 1/41];
p1 = [1.98 1.96 1.94 1.92 1.9];
DiffCoef0 = @(x,y) 0.1;
DiffCoef = {@(x,y) (2 + p1(1)* sin(2*pi*x/epsilon(1)) ) / ( 2 - p1(1)* cos(2*pi*y/epsilon(1)) ),   ...,
            @(x,y) (2 + p1(2)* sin(2*pi*(x+y)/sqrt(2)/epsilon(2)) ) / ( 2 - p1(2)* sin(2*pi*(x-y)/sqrt(2)/epsilon(2)) ), ...,
            @(x,y) (2 + p1(3)* cos(2*pi*(x-0.5)/epsilon(3)) ) / ( 2 - p1(3)* cos(2*pi*(y-0.5)/epsilon(3)) ), ...,
            @(x,y) (2 + p1(4)* cos(2*pi*(x-y)/sqrt(2)/epsilon(4)) ) / ( 2 - p1(4)* sin(2*pi*(x+y)/sqrt(2)/epsilon(4)) ), ...,
            @(x,y) (2 + p1(5)* cos(2*pi*(2*x-y)/sqrt(5)/epsilon(5)) ) / ( 2 - p1(5)* sin(2*pi*(x+2*y)/sqrt(5)/epsilon(5)) ) };
A0 = StiffMat2D_sparse(DiffCoef0,p,t);
A1 = StiffMat2D_sparse(DiffCoef{1},p,t);
A2 = StiffMat2D_sparse(DiffCoef{2},p,t);
A3 = StiffMat2D_sparse(DiffCoef{3},p,t);
A4 = StiffMat2D_sparse(DiffCoef{4},p,t);
A5 = StiffMat2D_sparse(DiffCoef{5},p,t);

%  right hand side Dirac-delta function 'f' in eq.(**)
RightF = @(x,y)  ( sin(pi*2*x) * cos(pi*2*y) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
b1 = LoadVec2D_sparse(p,t,RightF);

%% Monte Carlo sampling (compute low dimensional basis)
K = 2000;
U = zeros(size(p,1),K);
tic;
parfor k = 1:K
    xi = rand(5,1);
    u1 = (A0+A1*xi(1)+A2*xi(2)+A3*xi(3)+A4*xi(4)+A5*xi(5) + R1) \ (b1 + r1);
    U(:,k) = u1;

    if (mod(k,100)==0)
        fprintf('k=%d\n',k); end;
end;
toc;


%% global problem (Galerkin)

accweight = 1 - (5:-1:1).^2*1e-6;
EffR = zeros(size(accweight));
L2err_mean = zeros(size(accweight));
H1err_mean = zeros(size(accweight));
Engerr_mean = zeros(size(accweight));
L2proj_mean = zeros(size(accweight));
H1proj_mean = zeros(size(accweight));
Engproj_mean = zeros(size(accweight));
L2err_std = zeros(size(accweight));
H1err_std = zeros(size(accweight));
Engerr_std = zeros(size(accweight));
L2proj_std = zeros(size(accweight));
H1proj_std = zeros(size(accweight));
Engproj_std = zeros(size(accweight));
K2 = 200;

for i = 1:length(accweight)
    [R,www,S,V1] = Effective_rank(U'* M1 * U,accweight(i));
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    Phi = (U * V1(:,1:R)) ./ repmat(sqrt(S(1:R))',size(U,1),1);
    
    
    L2err = [];
    L2proj = [];
    Engerr = [];
    Engproj = [];
    H1err = [];
    H1proj = [];
    parfor k = 1:K2
        % real solution
        xi = rand(5,1);
        A_stiff = A0+A1*xi(1)+A2*xi(2)+A3*xi(3)+A4*xi(4)+A5*xi(5);
        u1 = (A_stiff + R1) \ (b1 + r1);

        % 'projection' and 'Galerkin' solution
        u1_L2proj = Phi * ((Phi' * M1 * Phi) \ (Phi' * M1 * u1));
        u1_H1proj = Phi * ((Phi' * H1 * Phi) \ (Phi' * H1 * u1));
        u1_Engproj = Phi * ((Phi' * A_stiff * Phi) \ (Phi' * A_stiff * u1));
        u1_POD = Phi * ( (Phi' * (A_stiff + R1) * Phi) \ (Phi' * (b1 + r1)) );
        
        % errors
        L2err(k) = sqrt( ((u1_POD-u1)'*M1*(u1_POD-u1)) / (u1'*M1*u1) );
        Engerr(k) =  sqrt( ((u1_POD-u1)'*A_stiff*(u1_POD-u1)) / (u1'*A_stiff*u1) );
        H1err(k) = sqrt( ((u1_POD-u1)'*H1*(u1_POD-u1)) / (u1'*H1*u1) );

        L2proj(k) = sqrt( ((u1_L2proj-u1)'*M1*(u1_L2proj-u1)) / (u1'*M1*u1) );
        Engproj(k) =  sqrt( ((u1_Engproj-u1)'*A_stiff*(u1_Engproj-u1)) / (u1'*A_stiff*u1) );
        H1proj(k) =  sqrt( ((u1_H1proj-u1)'*H1*(u1_H1proj-u1)) / (u1'*H1*u1) );
    end;
    
    % save 'mean' and 'std' of testing/proj errors,
    % ‘L2,H1,Eng’proj其实都是解的L2投影 的各类误差，并不是各范数空间的投影误差
    EffR(i) = R;
    accweight(i) = www;
    L2err_mean(i) = mean(L2err);
    H1err_mean(i) = mean(H1err);
    Engerr_mean(i) = mean(Engerr);
    L2proj_mean(i) = mean(L2proj);
    H1proj_mean(i) = mean(H1proj);
    Engproj_mean(i) = mean(Engproj);
    L2err_std(i) = std(L2err);
    H1err_std(i) = std(H1err);
    Engerr_std(i) = std(Engerr);
    L2proj_std(i) = std(L2proj);
    H1proj_std(i) = std(H1proj);
    Engproj_std(i) = std(Engproj);
end;

save('Example1_5rvs_global_data','U','M1','H1','-v7.3');
save('Example1_5rvs_global_Galerkin','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','Engerr_std','L2proj_mean','H1proj_mean','Engproj_mean','L2proj_std','H1proj_std','Engproj_std','K2','-v7.3');



%% local problem (dense interpolation)
U_loc = U(loc_idx,:);
H1_loc = H1(loc_idx,loc_idx);
M1_loc = M1(loc_idx,loc_idx);

accweight = 1 - (5:-1:1).^2*1e-6;
EffR = zeros(size(accweight));
L2err_mean = zeros(size(accweight));
H1err_mean = zeros(size(accweight));
Engerr_mean = zeros(size(accweight));
L2proj_mean = zeros(size(accweight));
H1proj_mean = zeros(size(accweight));
Engproj_mean = zeros(size(accweight));
L2err_std = zeros(size(accweight));
H1err_std = zeros(size(accweight));
Engerr_std = zeros(size(accweight));
L2proj_std = zeros(size(accweight));
H1proj_std = zeros(size(accweight));
Engproj_std = zeros(size(accweight));
K2 = 200;

for i = 1:length(accweight)
    [R,www,S_loc,V1] = Effective_rank(U_loc'* M1_loc * U_loc,accweight(i));
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    Phi = (U_loc * V1(:,1:R)) ./ repmat(sqrt(S_loc(1:R))',size(U_loc,1),1);
    
    % training set (for interp)
    n = 9;
    training_size = n^5;
    Xinput = zeros(5,training_size);
    Youtput = zeros(R,training_size);
    parfor k = 1:training_size
        xi = rand(5,1);
        if (k<= training_size)
            xi = ([mod(k-1,n)+1 mod(ceil(k/n)-1,n)+1 mod(ceil(k/(n^2))-1,n)+1 mod(ceil(k/(n^3))-1,n)+1 mod(ceil(k/(n^4))-1,n)+1]' /(n-1) - 1/(n-1));
        end;

        % real solution
        A_stiff = A0+A1*xi(1)+A2*xi(2)+A3*xi(3)+A4*xi(4)+A5*xi(5);
        u1 = (A_stiff + R1) \ (b1 + r1);
        u1 = u1(loc_idx);

        % 'projection' solution
        Youtput(:,k) = (Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1);
        Xinput(:,k) = xi;
    end;
    
    % training (dense interpolation)
    F = cell(1,R);
    for r = 1:R
        F{r} = griddedInterpolant(reshape(Xinput(1,:),[n n n n n]),reshape(Xinput(2,:),[n n n n n]), ...,
            reshape(Xinput(3,:),[n n n n n]),reshape(Xinput(4,:),[n n n n n]),reshape(Xinput(5,:),[n n n n n]), ...,
            reshape(Youtput(r,:),[n n n n n]),'spline');
    end;
    
    % testing 
    K2 = 200;
    L2err = [];
    L2proj = [];
    Engerr = [];
    Engproj = [];
    H1err = [];
    H1proj = [];
    parfor k = 1:K2
        % real solution
        xi = rand(5,1);
        A_stiff = A0+A1*xi(1)+A2*xi(2)+A3*xi(3)+A4*xi(4)+A5*xi(5);
        u1 = (A_stiff + R1) \ (b1 + r1);
        u1 = u1(loc_idx);

        % 'projection' and 'dense interp' solution
        u1_L2proj = Phi * ((Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1));
        u1_H1proj = Phi * ((Phi' * H1_loc * Phi) \ (Phi' * H1_loc * u1));
        u1_Engproj = Phi * ((Phi' * A_stiff(loc_idx,loc_idx) * Phi) \ (Phi' * A_stiff(loc_idx,loc_idx) * u1));
        Yc = zeros(R,1);
        for r = 1:R
            Yc(r) = F{r}(xi(1),xi(2),xi(3),xi(4),xi(5));
        end;
        u1_pred = Phi * Yc;
        
        % errors
        L2err(k) = sqrt( ((u1_pred-u1)'*M1_loc*(u1_pred-u1)) / (u1'*M1_loc*u1) );
        Engerr(k) =  sqrt( ((u1_pred-u1)'*A_stiff(loc_idx,loc_idx)*(u1_pred-u1)) / (u1'*A_stiff(loc_idx,loc_idx)*u1) );
        H1err(k) = sqrt( ((u1_pred-u1)'*H1_loc*(u1_pred-u1)) / (u1'*H1_loc*u1) );

        L2proj(k) = sqrt( ((u1_L2proj-u1)'*M1_loc*(u1_L2proj-u1)) / (u1'*M1_loc*u1) );
        Engproj(k) =  sqrt( ((u1_Engproj-u1)'*A_stiff(loc_idx,loc_idx)*(u1_Engproj-u1)) / (u1'*A_stiff(loc_idx,loc_idx)*u1) );
        H1proj(k) =  sqrt( ((u1_H1proj-u1)'*H1_loc*(u1_H1proj-u1)) / (u1'*H1_loc*u1) );
        
    end;
    
    % save 'mean' and 'std' of testing/proj errors,
    % ‘L2,H1,Eng’proj其实都是解的L2投影 的各类误差，并不是各范数空间的投影误差
    EffR(i) = R;
    accweight(i) = www;
    L2err_mean(i) = mean(L2err);
    H1err_mean(i) = mean(H1err);
    Engerr_mean(i) = mean(Engerr);
    L2proj_mean(i) = mean(L2proj);
    H1proj_mean(i) = mean(H1proj);
    Engproj_mean(i) = mean(Engproj);
    L2err_std(i) = std(L2err);
    H1err_std(i) = std(H1err);
    Engerr_std(i) = mean(Engerr);
    L2proj_std(i) = std(L2proj);
    H1proj_std(i) = std(H1proj);
    Engproj_std(i) = mean(Engproj);
end;

save('Example1_5rvs_local_data','loc_idx','M1_loc','H1_loc','-v7.3');
save('Example1_5rvs_local_denseInterp','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','L2proj_mean','H1proj_mean','Engproj_mean','Engerr_std','L2proj_std','H1proj_std','Engproj_std','training_size','K2','-v7.3');



