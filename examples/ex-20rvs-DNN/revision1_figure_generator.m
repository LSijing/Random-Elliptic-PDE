
clear
load('revision1_Ex4_testingsize.mat')
%% R = 5
idx = 1;
Nt = length(t_plot);
training_loss_plot = training_loss_array_cell(1,gap:gap:end,idx);
testing_loss_plot = mean(testing_loss_array_cell(:,:,idx),2);
L2err_mean = mean(L2err_array_cell(:,:,idx),2);
L2proj_mean = repmat(mean(L2proj_array_cell(:,:,idx)),Nt,1);
H1err_mean = mean(H1err_array_cell(:,:,idx),2);
H1proj_mean = repmat(mean(L2proj_array_cell(:,:,idx)),Nt,1);


figure(401);
hold on;
plot(t_plot,training_loss_plot,'b--^','markersize',4);
plot(t_plot,testing_loss_plot,'r--*','markersize',4);
legend1 = legend('training loss','testing loss');
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('value of loss function','fontsize',15);
set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_loss_5','-dpdf');
close
%}

figure(402);
hold on;
line1 = plot(t_plot,L2err_mean,'r--*','markersize',4);
line2 = plot(t_plot,L2proj_mean,'b--','markersize',4);
set(gca,'YTickMode','manual','YTick',0:0.05:0.6);
ylim([0,0.55]);
legend1 = legend([line1 line2],{'testing error','projection error'});
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('relative L^2 error','fontsize',15);
%set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_L2err_5','-dpdf');
close;
%}



figure(403);
hold on;
line1 = plot(t_plot,H1err_mean,'r--*','markersize',4);
line2 = plot(t_plot,H1proj_mean,'b--','markersize',4);
set(gca,'YTickMode','manual','YTick',0:0.05:0.6);
ylim([0,0.55])
legend1 = legend([line1 line2],{'testing error','projection error'});
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('relative H^1 error','fontsize',15);
%set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_H1err_5','-dpdf');
close;
%}


%% R = 10
clear
load('revision1_Ex4_testingsize.mat')
idx = 2;
Nt = length(t_plot);
training_loss_plot = training_loss_array_cell(1,gap:gap:end,idx);
testing_loss_plot = mean(testing_loss_array_cell(:,:,idx),2);
L2err_mean = mean(L2err_array_cell(:,:,idx),2);
L2proj_mean = repmat(mean(L2proj_array_cell(:,:,idx)),Nt,1);
H1err_mean = mean(H1err_array_cell(:,:,idx),2);
H1proj_mean = repmat(mean(L2proj_array_cell(:,:,idx)),Nt,1);


figure(401);
hold on;
plot(t_plot,training_loss_plot,'b--^','markersize',4);
plot(t_plot,testing_loss_plot,'r--*','markersize',4);
legend1 = legend('training loss','testing loss');
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('value of loss function','fontsize',15);
set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_loss_10','-dpdf');
close
%}

figure(402);
hold on;
line1 = plot(t_plot,L2err_mean,'r--*','markersize',4);
line2 = plot(t_plot,L2proj_mean,'b--','markersize',4);
% set(gca,'YTickMode','manual','YTick',0:0.05:0.6);
% ylim([0,0.55]);
legend1 = legend([line1 line2],{'testing error','projection error'});
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('relative L^2 error','fontsize',15);
%set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_L2err_10','-dpdf');
close;
%}



figure(403);
hold on;
line1 = plot(t_plot,H1err_mean,'r--*','markersize',4);
line2 = plot(t_plot,H1proj_mean,'b--','markersize',4);
% set(gca,'YTickMode','manual','YTick',0:0.05:0.6);
% ylim([0,0.55])
legend1 = legend([line1 line2],{'testing error','projection error'});
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('relative H^1 error','fontsize',15);
%set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_H1err_10','-dpdf');
close;
%}

%% R = 20
clear
load('revision1_Ex4_testingsize.mat')
idx = 3;
Nt = length(t_plot);
training_loss_plot = training_loss_array_cell(1,gap:gap:end,idx);
testing_loss_plot = mean(testing_loss_array_cell(:,:,idx),2);
L2err_mean = mean(L2err_array_cell(:,:,idx),2);
L2proj_mean = repmat(mean(L2proj_array_cell(:,:,idx)),Nt,1);
H1err_mean = mean(H1err_array_cell(:,:,idx),2);
H1proj_mean = repmat(mean(L2proj_array_cell(:,:,idx)),Nt,1);


figure(401);
hold on;
plot(t_plot,training_loss_plot,'b--^','markersize',4);
plot(t_plot,testing_loss_plot,'r--*','markersize',4);
legend1 = legend('training loss','testing loss');
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('value of loss function','fontsize',15);
set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_loss_20','-dpdf');
close
%}

figure(402);
hold on;
line1 = plot(t_plot,L2err_mean,'r--*','markersize',4);
line2 = plot(t_plot,L2proj_mean,'b--','markersize',4);
% set(gca,'YTickMode','manual','YTick',0:0.05:0.6);
% ylim([0,0.55]);
legend1 = legend([line1 line2],{'testing error','projection error'});
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('relative L^2 error','fontsize',15);
%set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_L2err_20','-dpdf');
close;
%}



figure(403);
hold on;
line1 = plot(t_plot,H1err_mean,'r--*','markersize',4);
line2 = plot(t_plot,H1proj_mean,'b--','markersize',4);
set(gca,'YTickMode','manual','YTick',0:0.05:0.6);
ylim([0,0.55])
legend1 = legend([line1 line2],{'testing error','projection error'});
set(legend1,'Fontsize',14);
xlabel('Number of training','fontsize',15);
ylabel('relative H^1 error','fontsize',15);
%set(gca,'Yscale','log');
box on;

%{
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'revision1_ex4_local_H1err_20','-dpdf');
close;
%}
