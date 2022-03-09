clear all
%% R and C initialization 
RawD = csvread('Case600.csv',2,1);
ts=120;%s
n=7;% number of state variables
m=12;% number of input signals
u=zeros(length(RawD),m);   
     
u(:,1)=RawD(:,1);     % ta 
u(:,2)=RawD(:,2);     % tg
u(:,3)=RawD(:,8)+RawD(:,11)+RawD(:,14)+RawD(:,17);    % qsolew: External solar irradiation.
u(:,4)=RawD(:,7)+ RawD(:,10)+RawD(:,13)+RawD(:,16);    % qg,r,ew: internal solar irradiation +radative internal heat gain.
u(:,5)=RawD(:,23);    % qsolc: External solar irradiation for ceiling.
u(:,6)=RawD(:,22);    % qg,r,c: radative internal heat gain.
u(:,7)=RawD(:,19);    % qg,r,f:  radative internal heat gain.
u(:,8)=RawD(:,3);    % qgc Internal heat gains convective heat gain.
u(:,9)=RawD(:,4)+RawD(:,5); % transmitted solar radiation 
u(:,10)=-RawD(:,27)/(ts); % infiltration
u(:,11)=RawD(:,26);     % tz
u(1,12)=0;% tz-tz-1 first data point
for j=2:length(RawD)
u(j,12)=(u(j,11)-u(j-1,11))/ts;     % (tz,k-tz,k-1)/ts
end
u=u';
ym=(RawD(:,28)-RawD(:,29))/(ts);

ndata = size(u,1); 
nstart=1;%4380*30;
ntrain=168*30; %4380*30+168*30;
ntest=168*30;
%ntrain=4380; %floor(0.1*ndata);
%ntest=ndata-ntrain;
utrain=u(1:m,nstart:ntrain);
ymtrain=ym(nstart:ntrain);
utest=u(1:m,ntrain+1:ntrain+ntest);
ymtest=ym(ntrain+1:ntrain+ntest);
[x0]=construction();
save ('nonlinear.mat', 'utrain', 'ymtrain', 'n','m','nstart','ntrain','ts');

% R_e1 = 0.00018;
% R_e2 = 0.019575;
% R_e3 = 0.001157;
% C_e1 = 11836.38;
% C_e2 = 11836.38;
% 
% R_c1 = 0.000216;
% R_c2 = 0.023491;
% R_c3 = 0.001389;
% C_c1 = 9863.369;
% C_c2 = 9863.369;
% 
% R_f1 = 0.000586;
% R_f2=0.001389;
% C_f1= 39473.22;
%R_f2 = 0.000334;
%R_f3 = 0.001389;
%C_f1 = 19736.61; 
%C_f2 = 19736.61; 

AR=zeros(n,n);
BR=zeros(n,m);
CR=zeros(1,n);
DR=zeros(1,m);
%x0=[R_e1 R_e2 R_e3 C_e1 C_e2 R_c1 R_c2 R_c3 C_c1 C_c2 R_f1 R_f2 R_f3 C_f1 C_f2];
%x0=[R_e1 R_e2 R_e3 C_e1 C_e2 R_c1 R_c2 R_c3 C_c1 C_c2 R_f1 R_f2 C_f1];
[A B C D]=M_ssMatrix(x0,n,m);
[x]=lsqRCIdentification(x0);
save ('pso_matlab.mat', 'x');
%[AR BR CR DR]=M_ssMatrix(x,n,m);
%ytest_org=ss2tf(A,B,C,D,n,utest,ymtest);
ytest=SS_function(x,utest,ymtest,n,m,ts);
%ytrain_org=ss2tf(A,B,C,D,n,utrain,ymtrain);
ytrain=SS_function(x,utrain,ymtrain,n,m,ts);
%ytrain=ss2tf(AR,BR,CR,DR,n,utrain,ymtrain);
%[R CVrmse RMSE NMBE]=stat(ytest_org, ymtest);
[R2 CVrmse2 RMSE2 NMBE2]=stat(ytest, ymtest);
prediction_error_test = goodnessOfFit(ymtest,ytest,'NRMSE');
figure();
plot(1:ntest,ytest,1:ntest,ymtest);
legend('Predicted','Measured');
str = sprintf('Test data NRMSE = %f',prediction_error_test);
title(str);
disp(prediction_error_test);
%[R3 CVrmse3 RMSE3 NMBE3]=stat(ytrain_org, ymtrain);
[R4 CVrmse4 RMSE4 NMBE4]=stat(ytrain, ymtrain);
prediction_error_train = goodnessOfFit(ymtrain,ytrain,'NRMSE');
figure();
plot(1:ntrain-nstart+1,ytrain,1:ntrain-nstart+1,ymtrain);
legend('Predicted','Measured');
str = sprintf('Test data NRMSE = %f',prediction_error_train);
title(str);
disp(prediction_error_train);
function [R_sq CVrmse RMSE NMBE]=stat(predict, measure)
R_sq= 1- sum((predict-measure).^2)/sum((predict-mean(measure)).^2);
CVrmse=sqrt(sum((predict-measure).^2)/size(measure,1))/mean(measure);
RMSE=sqrt(sum((predict-measure).^2)/size(measure,1));
NMBE=sum(predict-measure)/size(measure,1)/mean(measure);
end
