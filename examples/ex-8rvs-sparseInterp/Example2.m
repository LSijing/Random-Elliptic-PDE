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

% setting of RHS function
RightF = @(x,y)  ( cos(pi*2*x) * sin(pi*2*y) )* (x>=Do_2(1)) * (x<=Do_2(1)+Do_2(3)) * (y>=Do_2(2)) * (y<=Do_2(2)+Do_2(4));
b1 = LoadVec2D_sparse(p,t,RightF); 

% setting of Diffusion Coefficient
epsilon = [ 1/43 1/41 1/47 1/29 1/37 1/31 1/53 1/35];


%% Monte Carlo sampling (compute low dimensional basis)
K = 2000;
U = zeros(size(p,1),K);
time_record = zeros(10,1);
time_describe = {'offline K samples for basis extraction','Global basis extraction,5','Global fine solution,5*K2','Global Galerkin solution,5*K2', ...,
                    'Local basis extraction,5','offline sampling for Local mapping,5*training_size','Local mapping training,5','Local fine solution,5*K2','Local mapping solution,5*K2'};
tic;
parfor k = 1:K
    xi = rand(8,1)*2-1;
    DiffCoef = @(x,y) exp( sin(8/9*2*pi*x/epsilon(1)) * cos(1/9*2*pi*y/epsilon(1))* xi(1)/2     + sin(7/9*2*pi*x/epsilon(2)) * cos(2/9*2*pi*y/epsilon(2)) * xi(2)/2 ...,
                        + sin(6/9*2*pi*x/epsilon(3)) * cos(3/9*2*pi*y/epsilon(3))* xi(3)/2     + sin(5/9*2*pi*x/epsilon(4)) * cos(4/9*2*pi*y/epsilon(4)) * xi(4)/2 ...,
                        + sin(4/9*2*pi*x/epsilon(5)) * cos(5/9*2*pi*y/epsilon(5))* xi(5)/2     + sin(3/9*2*pi*x/epsilon(6)) * cos(6/9*2*pi*y/epsilon(6)) * xi(6)/2 ...,
                        + sin(2/9*2*pi*x/epsilon(7)) * cos(7/9*2*pi*y/epsilon(7))* xi(7)/2     + sin(1/9*2*pi*x/epsilon(8)) * cos(8/9*2*pi*y/epsilon(8)) * xi(8)/2 );
                
    A1 = StiffMat2D_sparse(DiffCoef,p,t);
    u1 = (A1 + R1) \ (b1 + r1);
    U(:,k) = u1;
    
    if (mod(k,100)==0)
        fprintf('k=%d\n',k); end;
end;
time_record(1) = time_record(1) + toc;


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
    tic;
    [R,www,S,V1] = Effective_rank(U'* M1 * U,accweight(i));
    Phi = (U * V1(:,1:R)) ./ repmat(sqrt(S(1:R))',size(U,1),1);
    time_record(2) = time_record(2) + toc;
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    
    L2err = [];
    L2proj = [];
    Engerr = [];
    Engproj = [];
    H1err = [];
    H1proj = [];
    parfor k = 1:K2
        % real solution
        xi = rand(8,1)*2-1;
        DiffCoef = @(x,y) exp( sin(8/9*2*pi*x/epsilon(1)) * cos(1/9*2*pi*y/epsilon(1))* xi(1)/2     + sin(7/9*2*pi*x/epsilon(2)) * cos(2/9*2*pi*y/epsilon(2)) * xi(2)/2 ...,
                            + sin(6/9*2*pi*x/epsilon(3)) * cos(3/9*2*pi*y/epsilon(3))* xi(3)/2     + sin(5/9*2*pi*x/epsilon(4)) * cos(4/9*2*pi*y/epsilon(4)) * xi(4)/2 ...,
                            + sin(4/9*2*pi*x/epsilon(5)) * cos(5/9*2*pi*y/epsilon(5))* xi(5)/2     + sin(3/9*2*pi*x/epsilon(6)) * cos(6/9*2*pi*y/epsilon(6)) * xi(6)/2 ...,
                            + sin(2/9*2*pi*x/epsilon(7)) * cos(7/9*2*pi*y/epsilon(7))* xi(7)/2     + sin(1/9*2*pi*x/epsilon(8)) * cos(8/9*2*pi*y/epsilon(8)) * xi(8)/2 );
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

save('Example2_8rvs_global_data','U','M1','H1','-v7.3');
save('Example2_8rvs_global_Galerkin','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','Engerr_std','L2proj_mean','H1proj_mean','Engproj_mean','L2proj_std','H1proj_std','Engproj_std','K2','-v7.3');



%% local problem (sparse interpolation)

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
    tic;
    [R,www,S_loc,V1] = Effective_rank(U_loc'* M1_loc * U_loc,accweight(i));
    Phi = (U_loc * V1(:,1:R)) ./ repmat(sqrt(S_loc(1:R))',size(U_loc,1),1);
    time_record(5) = time_record(5) + toc;
    fprintf('Effective rank:%d, accumulated weights:%.4f%%\n',R,www*100);
    
    % training set (for sparse interp)
    tic;
    [Xinput, weights] = nwspgr('KPU', 8, 5);
    Xinput = Xinput'*2-1;
    training_size = size(Xinput,2);
    Youtput = zeros(R,training_size);
    Indexset = PolyChaos_indexset(8,4);
    
    parfor k = 1:training_size
        xi = Xinput(:,k);
        DiffCoef = @(x,y) exp( sin(8/9*2*pi*x/epsilon(1)) * cos(1/9*2*pi*y/epsilon(1))* xi(1)/2     + sin(7/9*2*pi*x/epsilon(2)) * cos(2/9*2*pi*y/epsilon(2)) * xi(2)/2 ...,
                            + sin(6/9*2*pi*x/epsilon(3)) * cos(3/9*2*pi*y/epsilon(3))* xi(3)/2     + sin(5/9*2*pi*x/epsilon(4)) * cos(4/9*2*pi*y/epsilon(4)) * xi(4)/2 ...,
                            + sin(4/9*2*pi*x/epsilon(5)) * cos(5/9*2*pi*y/epsilon(5))* xi(5)/2     + sin(3/9*2*pi*x/epsilon(6)) * cos(6/9*2*pi*y/epsilon(6)) * xi(6)/2 ...,
                            + sin(2/9*2*pi*x/epsilon(7)) * cos(7/9*2*pi*y/epsilon(7))* xi(7)/2     + sin(1/9*2*pi*x/epsilon(8)) * cos(8/9*2*pi*y/epsilon(8)) * xi(8)/2 );

        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        u1 = (A1 + R1) \ (b1 + r1);
        u1 = u1(loc_idx);

        % 'projection' solution
        Youtput(:,k) = (Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1);
    end;
    time_record(6) = time_record(6) + toc;
    
    % training (sparse interpolation)
    tic;
    alpha = zeros(size(Indexset,2),R);
    for r = 1:R
        parfor i = 1:size(Indexset,2)
            tmp = ones(training_size,1);
            for j = 1:8
                tmp = tmp .* Legendre1D(Indexset(j,i),Xinput(j,:)');
            end;
            alpha(i,r) = Youtput(r,:) * (weights .* tmp);
        end;
    end;
    time_record(7) = time_record(7) + toc;
    
    % testing 
    L2err = [];
    L2proj = [];
    Engerr = [];
    Engproj = [];
    H1err = [];
    H1proj = [];
    parfor k = 1:K2
        % real solution
        xi = rand(8,1)*2-1;
        DiffCoef = @(x,y) exp( sin(8/9*2*pi*x/epsilon(1)) * cos(1/9*2*pi*y/epsilon(1))* xi(1)/2     + sin(7/9*2*pi*x/epsilon(2)) * cos(2/9*2*pi*y/epsilon(2)) * xi(2)/2 ...,
                            + sin(6/9*2*pi*x/epsilon(3)) * cos(3/9*2*pi*y/epsilon(3))* xi(3)/2     + sin(5/9*2*pi*x/epsilon(4)) * cos(4/9*2*pi*y/epsilon(4)) * xi(4)/2 ...,
                            + sin(4/9*2*pi*x/epsilon(5)) * cos(5/9*2*pi*y/epsilon(5))* xi(5)/2     + sin(3/9*2*pi*x/epsilon(6)) * cos(6/9*2*pi*y/epsilon(6)) * xi(6)/2 ...,
                            + sin(2/9*2*pi*x/epsilon(7)) * cos(7/9*2*pi*y/epsilon(7))* xi(7)/2     + sin(1/9*2*pi*x/epsilon(8)) * cos(8/9*2*pi*y/epsilon(8)) * xi(8)/2 );

        A1 = StiffMat2D_sparse(DiffCoef,p,t);
        %tic;
        u1 = (A1 + R1) \ (b1 + r1);
        u1 = u1(loc_idx);
        %time_record(8) = time_record(8) + toc;

        % 'projection' and 'sparse interp' solution
        u1_L2proj = Phi * ((Phi' * M1_loc * Phi) \ (Phi' * M1_loc * u1));
        u1_H1proj = Phi * ((Phi' * H1_loc * Phi) \ (Phi' * H1_loc * u1));
        u1_Engproj = Phi * ((Phi' * A1(loc_idx,loc_idx) * Phi) \ (Phi' * A1(loc_idx,loc_idx) * u1));
        %tic;
        Yc = zeros(R,1);
        for r = 1:R
            tmp = ones(size(Indexset,2),1);
            for i = 1:size(Indexset,2)
                for j = 1:8
                    tmp(i) = tmp(i) .* Legendre1D(Indexset(j,i),xi(j));
                end;
            end;
            Yc(r) = alpha(:,r)' * tmp;
        end;
        u1_pred = Phi * Yc;
        %time_record(9) = time_record(9) + toc;
        
        % errors
        L2err(k) = sqrt( ((u1_pred-u1)'*M1_loc*(u1_pred-u1)) / (u1'*M1_loc*u1) );
        Engerr(k) =  sqrt( ((u1_pred-u1)'*A1(loc_idx,loc_idx)*(u1_pred-u1)) / (u1'*A1(loc_idx,loc_idx)*u1) );
        H1err(k) = sqrt( ((u1_pred-u1)'*H1_loc*(u1_pred-u1)) / (u1'*H1_loc*u1) );

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

save('Example2_8rvs_local_data','loc_idx','M1_loc','H1_loc','time_record','time_describe','-v7.3');
save('Example2_8rvs_local_sparseInterp','EffR','accweight','L2err_mean','H1err_mean','Engerr_mean','L2err_std','H1err_std','L2proj_mean','H1proj_mean','Engproj_mean','Engerr_std','L2proj_std','H1proj_std','Engproj_std','training_size','K2','-v7.3');



