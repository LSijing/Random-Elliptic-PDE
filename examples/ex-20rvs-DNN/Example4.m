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

% % setting of RHS function
% RightF = @(x,y)  ( cos(pi*2*x) * sin(pi*2*y) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
% b1 = LoadVec2D_sparse(p,t,RightF); 

% setting of Diffusion Coefficient
angle_coef = pi*(1:18)'/18;
epsilon = 1./(11:2:45)';
     
                       
%% Monte Carlo sampling (compute low dimensional basis)

K = 2000;
U = zeros(size(p,1),K);

tic;
parfor k = 1:K
    xi = rand(18,1)*2-1;
    DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/5 ) );
    theta = [rand(1,1)* Do_2(3) + Do_2(1) ; rand(1,1) * Do_2(4) + Do_2(2)];
    RightF = @(x,y) exp(-((x-theta(1))^2+(y-theta(2))^2) / 0.01^2 /2 ) / (2*pi*0.01^2)  ;
    b1 = LoadVec2D_sparse(p,t,RightF); 
    A1 = StiffMat2D_sparse(DiffCoef,p,t);
    
    u1 = (A1 + R1) \ (b1 + r1);
    U(:,k) = u1;
    
    if (mod(k,100)==0)
        fprintf('k=%d\n',k); end;
end;



%% global problem (Galerkin)

accweight = 1 - (10:-1:5).^2*1e-6;
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
    tic;
    [R,www,S,V1] = Effective_rank(U'* M1 * U,accweight(i));
    Phi = (U * V1(:,1:R)) ./ repmat(sqrt(S(1:R))',size(U,1),1);
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    
    L2err = [];
    L2proj = [];
    Engerr = [];
    Engproj = [];
    H1err = [];
    H1proj = [];
    parfor k = 1:K2
        % real solution
        xi = rand(18,1)*2-1;
        DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/5 ) );
        theta = [rand(1,1)* Do_2(3) + Do_2(1) ; rand(1,1) * Do_2(4) + Do_2(2)];
        RightF = @(x,y) exp(-((x-theta(1))^2+(y-theta(2))^2) / 0.01^2 /2 ) / (2*pi*0.01^2)  ;
        b1 = LoadVec2D_sparse(p,t,RightF); 
        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        %tic;
        u1 = (A1 + R1) \ (b1 + r1);
        %time_record(3) = time_record(3) + toc;
        
        % 'projection' and 'Galerkin' solution
        u1_L2proj = Phi * ((Phi' * M1 * Phi) \ (Phi' * M1 * u1));
        u1_H1proj = Phi * ((Phi' * H1 * Phi) \ (Phi' * H1 * u1));
        u1_Engproj = Phi * ((Phi' * A1 * Phi) \ (Phi' * A1 * u1));
        %tic;
        u1_POD = Phi * ( (Phi' * (A1 + R1) * Phi) \ (Phi' * (b1 + r1)) );
        %time_record(4) = time_record(4) + toc;
        
        % errors
        L2err(k) = sqrt( ((u1_POD-u1)'*M1*(u1_POD-u1)) / (u1'*M1*u1) );
        Engerr(k) =  sqrt( ((u1_POD-u1)'*A1*(u1_POD-u1)) / (u1'*A1*u1) );
        H1err(k) = sqrt( ((u1_POD-u1)'*H1*(u1_POD-u1)) / (u1'*H1*u1) );

        L2proj(k) = sqrt( ((u1_L2proj-u1)'*M1*(u1_L2proj-u1)) / (u1'*M1*u1) );
        Engproj(k) =  sqrt( ((u1_Engproj-u1)'*A1*(u1_Engproj-u1)) / (u1'*A1*u1) );
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

save('Example4_18rvs2_global_data','U','M1','H1','-v7.3');
save('Example4_18rvs2_global_Galerkin','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','Engerr_std','L2proj_mean','H1proj_mean','Engproj_mean','L2proj_std','H1proj_std','Engproj_std','K2','-v7.3');



%% local problem (DNN)

U_loc = U(loc_idx,:);
H1_loc = H1(loc_idx,loc_idx);
M1_loc = M1(loc_idx,loc_idx);

accweight = 1 - 1e-6;
EffR = zeros(size(accweight));
K2 = 200;

tic;
[R,www,S_loc,V1] = Effective_rank(U_loc'* M1_loc * U_loc,accweight);
Phi = (U_loc * V1(:,1:R)) ./ repmat(sqrt(S_loc(1:R))',size(U_loc,1),1);

fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);

% training set (for DNN)
tic;
training_size = 5000;
Xinput = zeros(20,training_size+K2);
Youtput = zeros(R,training_size+K2);
Up2 = zeros(size(U_loc,1),training_size+K2);
L2proj = [];
Engproj = [];
H1proj = [];

parfor k = 1:training_size + K2
    xi = rand(18,1)*2-1;
    DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/5 ) );
    theta = [rand(1,1)* Do_2(3) + Do_2(1) ; rand(1,1) * Do_2(4) + Do_2(2)];
    RightF = @(x,y) exp(-((x-theta(1))^2+(y-theta(2))^2) / 0.01^2 /2 ) / (2*pi*0.01^2)  ;
    b1 = LoadVec2D_sparse(p,t,RightF); 
    A1 = StiffMat2D_sparse(DiffCoef,p,t);
    u1 = (A1 + R1) \ (b1 + r1);
    u1 = u1(loc_idx);
    Up2(:,k) = u1;

    % 'projection' solution
    Xinput(:,k) = [xi;theta];
    Youtput(:,k) = (Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1);
    
    % 'projection' and 'Taylor expansion' solution
    u1_L2proj = Phi * ((Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1));
    u1_H1proj = Phi * ((Phi' * H1_loc * Phi) \ (Phi' * H1_loc * u1));
    u1_Engproj = Phi * ((Phi' * A1(loc_idx,loc_idx) * Phi) \ (Phi' * A1(loc_idx,loc_idx) * u1));


    % projection errors
    L2proj(k) = sqrt( ((u1_L2proj-u1)'*M1_loc*(u1_L2proj-u1)) / (u1'*M1_loc*u1) );
    Engproj(k) =  sqrt( ((u1_Engproj-u1)'*A1(loc_idx,loc_idx)*(u1_Engproj-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
    H1proj(k) =  sqrt( ((u1_H1proj-u1)'*H1_loc*(u1_H1proj-u1)) / (u1'*H1_loc*u1) );
    
end;




save('Example4_18rvs2_local_data','Up2','Xinput','Youtput','S_loc','Phi','loc_idx','M1_loc','H1_loc','R','www','L2proj','H1proj','Engproj','training_size','K2','-v7.3');




