clear all;
% 改动：1.accweight (5:-1:1) 改为 控制 EffR
% 2.不存global data 和 local data
% 3.global 问题用QR Sensor解，因为这是non parametric的问题
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
angle_coef = pi*(1:24)'/24;
epsilon = (2:25)'/100;

%% Monte Carlo sampling (compute tailored basis)
K = 2000;
U = zeros(size(p,1),K);
tic;
parfor k = 1:K
    xi = rand(24,1)*2-1;
    DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/6 ) );

    %  right hand side Dirac-delta function 'f' in eq.(**)
    theta = rand(4,1)*2;
    RightF = @(x,y)  ( sin(pi*(x*theta(1)+ 2*theta(2) )) * cos(pi*(y*theta(3)+2*theta(4) )) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
    b1 = LoadVec2D_sparse(p,t,RightF);
    
    A1 = StiffMat2D_sparse(DiffCoef,p,t);
    u1 = (A1 + R1) \ (b1 + r1);
    U(:,k) = u1;
    
    if (mod(k,100)==0)
        fprintf('k=%d\n',k); end;
end;
toc;


%% global problem (QR sensor)

EffR = sort([(1:66)*3-1, (2:66)*3-2]);
accweight = zeros(size(EffR));

L2err_mean = zeros(size(accweight));
H1err_mean = zeros(size(accweight));
Engerr_mean = zeros(size(accweight));
L2err_rdm_mean = zeros(size(accweight));
H1err_rdm_mean = zeros(size(accweight));
Engerr_rdm_mean = zeros(size(accweight));
L2proj_mean = zeros(size(accweight));
H1proj_mean = zeros(size(accweight));
Engproj_mean = zeros(size(accweight));
L2err_std = zeros(size(accweight));
H1err_std = zeros(size(accweight));
Engerr_std = zeros(size(accweight));
L2err_rdm_std = zeros(size(accweight));
H1err_rdm_std = zeros(size(accweight));
Engerr_rdm_std = zeros(size(accweight));
L2proj_std = zeros(size(accweight));
H1proj_std = zeros(size(accweight));
Engproj_std = zeros(size(accweight));
K2 = 20;

for i = 1:length(accweight)
    R = EffR(i);
    [www,S,V1] = Accumulated_sum(U'* M1 * U,R);
%     [R,www,S_loc,V1] = Effective_rank(U'* M1 * U,accweight(i));
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    Phi = (U * V1(:,1:R)) ./ repmat(sqrt(S(1:R))',size(U,1),1);
    
    
    L2err = [];
    L2err_rdm = [];
    L2proj = [];
    Engerr = [];
    Engerr_rdm = [];
    Engproj = [];
    H1err = [];
    H1err_rdm = [];
    H1proj = [];
    % sensor placement
    [Qm,Rm,pm] = qr(Phi','vector');
    Cm = ind2vec(pm(1:R),size(U,1))';
    Rcvr = Phi * inv(Cm * Phi);
    parfor k = 1:K2
        xi = rand(24,1)*2-1;
        DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/6 ) );

        %  right hand side Dirac-delta function 'f' in eq.(**)
        theta = rand(4,1)*2;
        RightF = @(x,y)  ( sin(pi*(x*theta(1)+ 2*theta(2) )) * cos(pi*(y*theta(3)+2*theta(4) )) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
        b1 = LoadVec2D_sparse(p,t,RightF);

        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        % real solution
        u1 = (A1 + R1) \ (b1 + r1);
        
        % QR sensor
        u1_QR = Rcvr * (Cm * u1);
        
        % random sensor
        C_rdm = ind2vec(randi(size(U,1),1,R),size(U,1))';
        u1_random = Phi * (inv(C_rdm * Phi) * (C_rdm * u1));
        
        % 'projection' solution
        u1_L2proj = Phi * ((Phi' * M1 * Phi) \ (Phi' * M1 * u1));
        u1_H1proj = Phi * ((Phi' * H1 * Phi) \ (Phi' * H1 * u1));
        u1_Engproj = Phi * ((Phi' * A1 * Phi) \ (Phi' * A1 * u1));
        
        % errors
        L2err(k) = sqrt( ((u1_QR-u1)'*M1*(u1_QR-u1)) / (u1'*M1*u1) );
        Engerr(k) =  sqrt( ((u1_QR-u1)'*A1*(u1_QR-u1)) / (u1'*A1*u1) );
        H1err(k) = sqrt( ((u1_QR-u1)'*H1*(u1_QR-u1)) / (u1'*H1*u1) );
        
        L2err_rdm(k) = sqrt( ((u1_random-u1)'*M1*(u1_random-u1)) / (u1'*M1*u1) );
        Engerr_rdm(k) =  sqrt( ((u1_random-u1)'*A1*(u1_random-u1)) / (u1'*A1*u1) );
        H1err_rdm(k) = sqrt( ((u1_random-u1)'*H1*(u1_random-u1)) / (u1'*H1*u1) );
        
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
    L2err_rdm_mean(i) = mean(L2err_rdm);
    H1err_rdm_mean(i) = mean(H1err_rdm);
    Engerr_rdm_mean(i) = mean(Engerr_rdm);
    L2proj_mean(i) = mean(L2proj);
    H1proj_mean(i) = mean(H1proj);
    Engproj_mean(i) = mean(Engproj);
    L2err_std(i) = std(L2err);
    H1err_std(i) = std(H1err);
    Engerr_std(i) = std(Engerr);
    L2err_rdm_std(i) = std(L2err_rdm);
    H1err_rdm_std(i) = std(H1err_rdm);
    Engerr_rdm_std(i) = std(Engerr_rdm);
    L2proj_std(i) = std(L2proj);
    H1proj_std(i) = std(H1proj);
    Engproj_std(i) = std(Engproj);
end;


save('Example3_24rvs4_global_QRsensor_v3','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','Engerr_std','L2proj_mean','H1proj_mean','Engproj_mean','L2proj_std','H1proj_std','Engproj_std','L2err_rdm_mean','H1err_rdm_mean','Engerr_rdm_mean','L2err_rdm_std','H1err_rdm_std','Engerr_rdm_std','K2','-v7.3');



%% local problem (QR sensor placement)

U_loc = U(loc_idx,:);
H1_loc = H1(loc_idx,loc_idx);
M1_loc = M1(loc_idx,loc_idx);

EffR = (2:50)*2-1;
accweight = zeros(size(EffR));

L2err_mean = zeros(size(accweight));
H1err_mean = zeros(size(accweight));
Engerr_mean = zeros(size(accweight));
L2err_rdm_mean = zeros(size(accweight));
H1err_rdm_mean = zeros(size(accweight));
Engerr_rdm_mean = zeros(size(accweight));
L2proj_mean = zeros(size(accweight));
H1proj_mean = zeros(size(accweight));
Engproj_mean = zeros(size(accweight));
L2err_std = zeros(size(accweight));
H1err_std = zeros(size(accweight));
Engerr_std = zeros(size(accweight));
L2err_rdm_std = zeros(size(accweight));
H1err_rdm_std = zeros(size(accweight));
Engerr_rdm_std = zeros(size(accweight));
L2proj_std = zeros(size(accweight));
H1proj_std = zeros(size(accweight));
Engproj_std = zeros(size(accweight));
K2 = 20;

for i = 1:length(accweight)
    R = EffR(i);
    [www,S_loc,V1] = Accumulated_sum(U_loc'* M1_loc * U_loc,R);
    %[R,www,S_loc,V1] = Effective_rank(U_loc'* M1_loc * U_loc,accweight(i));
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    Phi = (U_loc * V1(:,1:R)) ./ repmat(sqrt(S_loc(1:R))',size(U_loc,1),1);
    
    
    L2err = [];
    L2err_rdm = [];
    L2proj = [];
    Engerr = [];
    Engerr_rdm = [];
    Engproj = [];
    H1err = [];
    H1err_rdm = [];
    H1proj = [];
    % sensor placement
    [Qm,Rm,pm] = qr(Phi','vector');
    Cm = ind2vec(pm(1:R),size(U_loc,1))';
    Rcvr = Phi * inv(Cm * Phi);
    parfor k = 1:K2
        xi = rand(24,1)*2-1;
        DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/6 ) );

        %  right hand side Dirac-delta function 'f' in eq.(**)
        theta = rand(4,1)*2;
        RightF = @(x,y)  ( sin(pi*(x*theta(1)+ 2*theta(2) )) * cos(pi*(y*theta(3)+2*theta(4) )) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
        b1 = LoadVec2D_sparse(p,t,RightF);

        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        % real solution
        u1 = (A1 + R1) \ (b1 + r1);
        u1 = u1(loc_idx);

        % 'projection' and 'Galerkin' solution
        u1_L2proj = Phi * ((Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1));
        u1_H1proj = Phi * ((Phi' * H1_loc * Phi) \ (Phi' * H1_loc * u1));
        u1_Engproj = Phi * ((Phi' * A1(loc_idx,loc_idx) * Phi) \ (Phi' * A1(loc_idx,loc_idx) * u1));
        
        % QR sensor
        u1_QR = Rcvr * (Cm * u1);
        
        % random sensor
        C_rdm = ind2vec(randi(size(U_loc,1),1,R),size(U_loc,1))';
        u1_random = Phi * (inv(C_rdm * Phi) * (C_rdm * u1));

        % errors
        L2err(k) = sqrt( ((u1_QR-u1)'*M1_loc*(u1_QR-u1)) / (u1'*M1_loc*u1) );
        Engerr(k) =  sqrt( ((u1_QR-u1)'*A1(loc_idx,loc_idx)*(u1_QR-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
        H1err(k) = sqrt( ((u1_QR-u1)'*H1_loc*(u1_QR-u1)) / (u1'*H1_loc*u1) );

        L2err_rdm(k) = sqrt( ((u1_random-u1)'*M1_loc*(u1_random-u1)) / (u1'*M1_loc*u1) );
        Engerr_rdm(k) =  sqrt( ((u1_random-u1)'*A1(loc_idx,loc_idx)*(u1_random-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
        H1err_rdm(k) = sqrt( ((u1_random-u1)'*H1_loc*(u1_random-u1)) / (u1'*H1_loc*u1) );

        L2proj(k) = sqrt( ((u1_L2proj-u1)'*M1_loc*(u1_L2proj-u1)) / (u1'*M1_loc*u1) );
        Engproj(k) =  sqrt( ((u1_Engproj-u1)'*A1(loc_idx,loc_idx)*(u1_Engproj-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
        H1proj(k) =  sqrt( ((u1_H1proj-u1)'*H1_loc*(u1_H1proj-u1)) / (u1'*H1_loc*u1) );
    end;
    
    % save 'mean' and 'std' of testing/proj errors,
    % ‘L2,H1,Eng’proj其实都是解的L2投影 的各类误差，并不是各范数空间的投影误差
    EffR(i) = R;
    accweight(i) = www;
    L2err_mean(i) = mean(L2err);
    H1err_mean(i) = mean(H1err);
    Engerr_mean(i) = mean(Engerr);
%     L2err_rdm = L2err_rdm(~isnan(L2err_rdm));
%     H1err_rdm = H1err_rdm(~isnan(H1err_rdm));
%     Engerr_rdm = Engerr_rdm(~isnan(Engerr_rdm));
    L2err_rdm_mean(i) = mean(L2err_rdm);
    H1err_rdm_mean(i) = mean(H1err_rdm);
    Engerr_rdm_mean(i) = mean(Engerr_rdm);
    L2proj_mean(i) = mean(L2proj);
    H1proj_mean(i) = mean(H1proj);
    Engproj_mean(i) = mean(Engproj);
    L2err_std(i) = std(L2err);
    H1err_std(i) = std(H1err);
    Engerr_std(i) = std(Engerr);
    L2err_rdm_std(i) = std(L2err_rdm);
    H1err_rdm_std(i) = std(H1err_rdm);
    Engerr_rdm_std(i) = std(Engerr_rdm);
    L2proj_std(i) = std(L2proj);
    H1proj_std(i) = std(H1proj);
    Engproj_std(i) = std(Engproj);
end;

%save('Example3_24rvs4_local_data','loc_idx','M1_loc','H1_loc','-v7.3');
save('Example3_24rvs4_local_QRsensor_v3','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','Engerr_std','L2err_rdm_mean','H1err_rdm_mean','Engerr_rdm_mean','L2err_rdm_std','H1err_rdm_std','Engerr_rdm_std','L2proj_mean','H1proj_mean','Engproj_mean','L2proj_std','H1proj_std','Engproj_std','K2','-v7.3');
%save('Example3_24rvs4_global_data','U','M1','H1','epsilon','-v7.3');


