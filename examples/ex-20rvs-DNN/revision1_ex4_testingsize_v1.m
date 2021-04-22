
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



%% online 
load('Example4_18rvs2_local_data');
load('revision1_Ex4_local_data.mat');
load('revision1_Ex4_onlineoutput5.mat');
gap = 500;
t_plot = training_time_array(gap:gap:end);
Nt = length(t_plot);
EffR = [5 10 20];

Ypred_array_cell = cell(size(EffR));
training_loss_array_cell = zeros([size(training_loss_array),length(EffR)]);
testing_loss_array_cell = zeros(Nt,K2,length(EffR));
L2err_array_cell = zeros(Nt,K2,length(EffR));
H1err_array_cell = zeros(Nt,K2,length(EffR));
L2proj_array_cell = zeros(1,K2,length(EffR));
H1proj_array_cell = zeros(1,K2,length(EffR));
for i = 1:length(EffR)
    load(sprintf('revision1_Ex4_onlineoutput%d.mat',EffR(i)));
    Ypred_array_cell{i} = Ypred_array;
    training_loss_array_cell(:,:,i) = training_loss_array;
end;
t_FEM = zeros(1,K2);


parfor k = 1:K2
    xi = Xonline(1:18,k);
    theta = Xonline(19:20,k);
    % real solution
    tic;
    DiffCoef = @(x,y) exp( sum (sin( 2*pi*(x*sin(angle_coef) + y*cos(angle_coef))./epsilon ) .* xi/5 ) );
    RightF = @(x,y) exp(-((x-theta(1))^2+(y-theta(2))^2) / 0.01^2 /2 ) / (2*pi*0.01^2)  ;
    b1 = LoadVec2D_sparse(p,t,RightF); 
    A1 = StiffMat2D_sparse(DiffCoef,p,t);
    u1 = (A1 + R1) \ (b1 + r1);
    u1 = u1(loc_idx);
    t_FEM(k) = toc;
    Utest = repmat(u1,1,Nt);
    
    tmp_loss = zeros(Nt,length(EffR));
    tmp_L2err = zeros(Nt,length(EffR));
    tmp_H1err = zeros(Nt,length(EffR));
    tmp_L2proj = zeros(1,length(EffR));
    tmp_H1proj = zeros(1,length(EffR));
    for i = 1:length(EffR)
        R = EffR(i);
        PhiR = Phi(:,1:R);
        % 'projection' coefficient and solutions
        Ytest = (PhiR' * M1_loc * PhiR) \ (PhiR' * M1_loc * u1);
        u1_L2proj = PhiR * Ytest;
        u1_H1proj = PhiR * ((PhiR' * H1_loc * PhiR) \ (PhiR' * H1_loc * u1));

        % loss and errors
        Ypred = Ypred_array_cell{i}(:,k:K2:(k+K2*(Nt-1)));
        Upred = PhiR * Ypred;
        tmp_loss(:,i) = sum((Ypred - repmat(Ytest,1,Nt)).^2)/R;
        tmp_L2err(:,i) = sqrt(  diag((Upred - Utest)'*M1_loc*(Upred - Utest)) / diag(u1'*M1_loc*u1)  );
        tmp_H1err(:,i) = sqrt(  diag((Upred - Utest)'*H1_loc*(Upred - Utest)) / diag(u1'*H1_loc*u1)  );
        tmp_L2proj(i) = sqrt(  diag((u1_L2proj - u1)'*M1_loc*(u1_L2proj - u1)) / diag(u1'*M1_loc*u1)  );
        tmp_H1proj(i) = sqrt(  diag((u1_H1proj - u1)'*H1_loc*(u1_H1proj - u1)) / diag(u1'*H1_loc*u1)  );
    end;
    
    testing_loss_array_cell(:,k,:) = tmp_loss;
    L2err_array_cell(:,k,:) = tmp_L2err;
    H1err_array_cell(:,k,:) = tmp_H1err; 
    L2proj_array_cell(:,k,:) = tmp_L2proj;
    H1proj_array_cell(:,k,:) = tmp_H1proj; 
    
    if (mod(k,round(K2/100))==0 )
        fprintf('k=%d\n',k/round(K2/100));
    end;
end;

save('revision1_Ex4_testingsize.mat','K2','t_init','t_FEM','L2err_array_cell','H1err_array_cell','L2proj_array_cell','H1proj_array_cell','t_plot','training_loss_array_cell','gap','testing_loss_array_cell','-v7.3');


Nt = length(t_plot);
i = 2;
figure(401);
hold on;
plot(t_plot,training_loss_array_cell(1,gap:gap:end,i),'b--^','markersize',4);
plot(t_plot,mean(testing_loss_array_cell(:,:,i),2),'r--*','markersize',4);
legend1 = legend('training loss','testing loss');
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('value of loss function','fontsize',15);
set(gca,'Yscale','log');
box on;

figure(402);
hold on;
plot(t_plot,mean(L2err_array_cell(:,:,i),2));
plot(t_plot,repmat(mean(L2proj_array_cell(:,:,i)),Nt,1));



figure(403);
hold on;
plot(t_plot,mean(H1err_array_cell(:,:,i),2));
plot(t_plot,repmat(mean(H1proj_array_cell(:,:,i)),Nt,1));

