clear all;

%% A pde solver for moving interface problem (interface with random place)
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

% setting of RHS function
RightF = @(x,y)  ( cos(pi*2*x) * sin(pi*2*y) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
b1 = LoadVec2D_sparse(p,t,RightF); 


% setting of Diffusion Coefficient
K0 = [1 3000];
% 坐下横坐标，左下纵坐标，宽度，高度
rectK0 = [ floor(0.3*N)*h  floor(0.1*N)*h  10*h  floor(0.8*N)*h ;
           floor(0.5*N)*h  floor(0.1*N)*h  10*h  floor(0.8*N)*h ;
           floor(0.7*N)*h  floor(0.1*N)*h  10*h  floor(0.8*N)*h ];

       
angle_coef1 = pi*(1:6)'/6;
epsilon1 = (2:7)'/100;
angle_coef2 = pi*( (1:6) + 0.5)'/6;
epsilon2 = (20:25)'/100;

                       
%% Monte Carlo sampling (compute low dimensional basis)

K = 2000;
load('Example5_12rvs_global_data');



%% local problem ( Taylor expansion )



U_loc = U(loc_idx,:);
H1_loc = H1(loc_idx,loc_idx);
M1_loc = M1(loc_idx,loc_idx);

% accweight = 1 - (10:-2:2).^2*1e-6;
% EffR = zeros(size(accweight));
EffR = 3:2:25;
accweight = zeros(size(EffR));
L2err_mean = zeros(size(accweight));
H1err_mean = zeros(size(accweight));
Engerr_mean = zeros(size(accweight));
Fluxerr_mean = zeros(size(accweight));
L2proj_mean = zeros(size(accweight));
H1proj_mean = zeros(size(accweight));
Engproj_mean = zeros(size(accweight));
Fluxproj_mean = zeros(size(accweight));
L2err_std = zeros(size(accweight));
H1err_std = zeros(size(accweight));
Engerr_std = zeros(size(accweight));
Fluxerr_std = zeros(size(accweight));
L2proj_std = zeros(size(accweight));
H1proj_std = zeros(size(accweight));
Engproj_std = zeros(size(accweight));
Fluxproj_std = zeros(size(accweight));
K2 = 200;

for i = 1:length(accweight)
    tic;
    R = EffR(i);
    [www,S_loc,V1] = Accumulated_sum(U_loc'* M1_loc * U_loc,R);
    Phi = (U_loc * V1(:,1:R)) ./ repmat(sqrt(S_loc(1:R))',size(U_loc,1),1);
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    
    % testing 
    L2err = [];
    L2proj = [];
    Engerr = [];
    Fluxerr = [];
    Engproj = [];
    H1err = [];
    H1proj = [];
    Fluxproj = [];

    % training set (for interp)
    tic;
    training_size = 5000;
    Youtput = zeros(R,training_size);
    
    parfor k = 1:training_size
        xi = rand(12,1)*2-1;
        DiffCoef = @(x,y) ( exp( sum (sin( 2*pi*(x*sin(angle_coef1) + y*cos(angle_coef1))./epsilon1 ) .* xi(1:6)/1.5 ) )  )...,
                            * (1 - any((x>=rectK0(:,1)) .* (x<=rectK0(:,1)+rectK0(:,3)) .* (y>=rectK0(:,2)) .* (y<=rectK0(:,2)+rectK0(:,4)) )  )...,
                        + ( exp( sum (sin( 2*pi*(x*sin(angle_coef2) + y*cos(angle_coef2))./epsilon2 ) .* xi(7:12)/1.5 ) )  )...,
                            * any((x>=rectK0(:,1)) .* (x<=rectK0(:,1)+rectK0(:,3)) .* (y>=rectK0(:,2)) .* (y<=rectK0(:,2)+rectK0(:,4)) );
        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        u1 = (A1 + R1) \ (b1 + r1);
        u1 = u1(loc_idx);

        % 'projection' solution
        Xinput(:,k) = xi;
        Youtput(:,k) = (Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1);
    end;

    % testing
    kdtree = KDTreeSearcher(Xinput');
    parfor k = 1:K2
        % real solution
        xi = rand(12,1)*2-1;
        DiffCoef = @(x,y) ( exp( sum (sin( 2*pi*(x*sin(angle_coef1) + y*cos(angle_coef1))./epsilon1 ) .* xi(1:6)/1.5 ) )  )...,
                            * (1 - any((x>=rectK0(:,1)) .* (x<=rectK0(:,1)+rectK0(:,3)) .* (y>=rectK0(:,2)) .* (y<=rectK0(:,2)+rectK0(:,4)) )  )...,
                        + ( exp( sum (sin( 2*pi*(x*sin(angle_coef2) + y*cos(angle_coef2))./epsilon2 ) .* xi(7:12)/1.5 ) )  )...,
                            * any((x>=rectK0(:,1)) .* (x<=rectK0(:,1)+rectK0(:,3)) .* (y>=rectK0(:,2)) .* (y<=rectK0(:,2)+rectK0(:,4)) );
        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        Flux1 = StiffMat2D_sparse(DiffCoef,p,t)
        u1 = (A1 + R1) \ (b1 + r1);
        u1 = u1(loc_idx);

        % 'projection' and 'Taylor expansion' solution
        u1_L2proj = Phi * ((Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1));
        u1_H1proj = Phi * ((Phi' * H1_loc * Phi) \ (Phi' * H1_loc * u1));
        u1_Engproj = Phi * ((Phi' * A1(loc_idx,loc_idx) * Phi) \ (Phi' * A1(loc_idx,loc_idx) * u1));
        u1_Fluxproj = Phi * ((Phi' * Flux1(loc_idx,loc_idx) * Phi) \ (Phi' * Flux1(loc_idx,loc_idx) * u1));
        
        Yc = zeros(R,1);
        [NNidx,NNdis] = knnsearch(kdtree,xi','K',20);
        for r = 1:R
            tmp = [ones(20,1),(repmat(xi',20,1) - Xinput(:,NNidx)')]\Youtput(r,NNidx)';
            Yc(r) = tmp(1);
        end;

        u1_pred = Phi * Yc;
        
        % errors
        L2err(k) = sqrt( ((u1_pred-u1)'*M1_loc*(u1_pred-u1)) / (u1'*M1_loc*u1) );
        Engerr(k) =  sqrt( ((u1_pred-u1)'*A1(loc_idx,loc_idx)*(u1_pred-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
        H1err(k) = sqrt( ((u1_pred-u1)'*H1_loc*(u1_pred-u1)) / (u1'*H1_loc*u1) );
        Fluxerr(k) =  sqrt( ((u1_pred-u1)'*Flux1(loc_idx,loc_idx)*(u1_pred-u1)) / (u1'*Flux1(loc_idx,loc_idx)*u1) );

        L2proj(k) = sqrt( ((u1_L2proj-u1)'*M1_loc*(u1_L2proj-u1)) / (u1'*M1_loc*u1) );
        Engproj(k) =  sqrt( ((u1_Engproj-u1)'*A1(loc_idx,loc_idx)*(u1_Engproj-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
        H1proj(k) =  sqrt( ((u1_H1proj-u1)'*H1_loc*(u1_H1proj-u1)) / (u1'*H1_loc*u1) );
        Fluxproj(k) =  sqrt( ((u1_Fluxproj-u1)'*Flux1(loc_idx,loc_idx)*(u1_Fluxproj-u1)) / (u1'*Flux1(loc_idx,loc_idx)*u1) );
        
    end;
    
    
    % save 'mean' and 'std' of testing/proj errors,
    % ‘L2,H1,Eng’proj其实都是解的L2投影 的各类误差，并不是各范数空间的投影误差
    EffR(i) = R;
    accweight(i) = www;
    L2err_mean(i) = mean(L2err);
    H1err_mean(i) = mean(H1err);
    Engerr_mean(i) = mean(Engerr);
    Fluxerr_mean(i) = mean(Fluxerr);
    L2proj_mean(i) = mean(L2proj);
    H1proj_mean(i) = mean(H1proj);
    Engproj_mean(i) = mean(Engproj);
    Fluxproj_mean(i) = mean(Fluxproj);
    L2err_std(i) = std(L2err);
    H1err_std(i) = std(H1err);
    Engerr_std(i) = std(Engerr);
    Fluxerr_std(i) = std(Fluxerr);
    L2proj_std(i) = std(L2proj);
    H1proj_std(i) = std(H1proj);
    Engproj_std(i) = std(Engproj);
    Fluxproj_std(i) = std(Fluxproj);
    
end;


save('Example5_12rvs_local_Taylorexpansion_V2','EffR','accweight','Fluxerr_mean','Fluxerr_std','Fluxproj_mean','Fluxproj_std','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','L2proj_mean','H1proj_mean','Engproj_mean','Engerr_std','L2proj_std','H1proj_std','Engproj_std','K2','-v7.3');



