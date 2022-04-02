%% Living Lab 1
clear
addpath('C:\Research\Paper\2019_paper_four\GMMR')
addpath(genpath('c:\Research\HMM'))
%% Load a dataset
m=2153; %first data point for training (April 1st)
n=4336; %3617; %end data point for training baseline
tm=4337;%3618; % first data point for testing (June 1st)
tn=4960; %end data point for testing
A = csvread('TT_out.csv',1,1);
data_var = A.'
data_tr = A(m:n,:); %Baselsie period  Janurary
data_tst = A(tm:tn,:);%A(n+1,:);%Testing         Feb-Dec
%% Asssigining variables
OAT=data_var(13,:);
nOAT=normalize(OAT);
SpaceT=(data_var(1,:)-0);
nSpaceT=normalize(SpaceT);
%OperationStatus=data_var(14,:);
RH=data_var(11,:);
nRH=normalize(RH);
FacadeT=data_var(12,:);
nFacadeT=normalize(FacadeT);
PCBControlV=data_var(6,:);
nPCBControlV=normalize(PCBControlV);
%SpaceH=data_var(2,:);
%nSpaceH=normalize(SpaceH);
HotValve=data_var(7,:);
nHotValve=normalize(HotValve);
ColdValve=data_var(8,:);
nColdValve=normalize(ColdValve);
%Pump_speed=data_var(19,:);
%nPump_speed = normalize(Pump_speed);

Press=data_var(21,:);
nPress = normalize(Press);

% AHU_SAT=data_var(10,:);
% nAHU_SAT=normalize(AHU_SAT);
%CHWST=data_var(9,:);
%nCHWST=normalize(CHWST);
Flow=data_var(15,:);
nFlow=normalize(Flow);
Tin=data_var(17,:);% supply water temperature
nTin=normalize(Tin);
%T_out=data_var(18,:);% return water temperature
%PCB_energy=0.063*5/9*4184/1000*(T_out-T_in).*Flow;
PCBenergy=data_var(25,:);%PCB cooling
nPCBenergy=normalize(PCBenergy);
% for i=1: length(PCB_energy)
%     if(PCB_energy(i)<=0)
%     PCB_energy(i)=0;
%     end
% end
% nPCBenergy=normalize(PCB_energy);
%Ele=data_var(8,:);
%nEle=normalize(Ele);
%Gas=data_var(29,:);
%nGas=normalize(Gas);
%T_1_energy = data_var(25,:);
 for i=1: length(PCBenergy)
    if(i<=1)
    T_1_energy(i)=PCBenergy(1);
    else
    T_1_energy(i)=PCBenergy(i-1);
    end
 end
nT_1_energy = normalize(T_1_energy);
%T_2_energy = data_var(25,:);
for i=1: length(PCBenergy)
    if(i<=2)
    T_2_energy(i)=PCBenergy(1);
    else
    T_2_energy(i)=PCBenergy(i-2);
    end
end
nT_2_energy = normalize(T_2_energy);
%T_3_energy = data_var(16,:);
for i=1: length(PCBenergy)
    if(i<=3)
    T_3_energy(i)=PCBenergy(1);
    else
    T_3_energy(i)=PCBenergy(i-3);
    end
end
nT_3_energy = normalize(T_3_energy);
for i=1: length(PCBenergy)
    if(i<=4)
    T_4_energy(i)=PCBenergy(1);
    else
    T_4_energy(i)=PCBenergy(i-4);
    end
end
nT_4_energy = normalize(T_4_energy);
for i=1: length(PCBenergy)
    if(i<=5)
    T_5_energy(i)=PCBenergy(1);
    else
    T_5_energy(i)=PCBenergy(i-5);
    end
end
nT_5_energy = normalize(T_5_energy);
for i=1: length(PCBenergy)
    if(i<=6)
    T_6_energy(i)=PCBenergy(1);
    else
    T_6_energy(i)=PCBenergy(i-6);
    end
end
nT_6_energy = normalize(T_6_energy);
% 
% %T_1_flow = data_var(17,:);
% for i=1: length(Flow)
%     if(i<=1)
%     T_1_flow(i)=Flow(1);
%     else
%     T_1_flow(i)=Flow(i-1);
%     end
% end
% nT_1_flow = normalize(T_1_flow);
% %T_2_flow = data_var(18,:);
% for i=1: length(Flow)
%     if(i<=2)
%     T_2_flow(i)=Flow(1);
%     else
%     T_2_flow(i)=Flow(i-2);
%     end
% end
% nT_2_flow = normalize(T_2_flow);
% %T_3_flow = data_var(19,:);
% for i=1: length(Flow)
%     if(i<=3)
%     T_3_flow(i)=Flow(1);
%     else
%     T_3_flow(i)=Flow(i-3);
%     end
% end
% nT_3_flow = normalize(T_3_flow);
% 
% for i=1: length(Flow)
%     if(i<=4)
%     T_4_flow(i)=Flow(1);
%     else
%     T_4_flow(i)=Flow(i-4);
%     end
% end
% for i=1: length(Flow)
%     if(i<=5)
%     T_5_flow(i)=Flow(1);
%     else
%     T_5_flow(i)=Flow(i-5);
%     end
% end
% for i=1: length(Flow)
%     if(i<=6)
%     T_6_flow(i)=Flow(1);
%     else
%     T_6_flow(i)=Flow(i-6);
%     end
% end
% for i=1: length(Flow)
%     if(i<=7)
%     T_7_flow(i)=Flow(1);
%     else
%     T_7_flow(i)=Flow(i-7);
%     end
% end
% for i=1: length(Flow)
%     if(i<=8)
%     T_8_flow(i)=Flow(1);
%     else
%     T_8_flow(i)=Flow(i-8);
%     end
% end
% nT_4_flow = normalize(T_4_flow);
% nT_5_flow = normalize(T_5_flow);
% nT_6_flow = normalize(T_6_flow);
% nT_7_flow = normalize(T_7_flow);
% nT_8_flow = normalize(T_8_flow);
% 
% for i=1: length(PCB_energy)
%     if(i<=4)
%     T_4_energy(i)=PCB_energy(1);
%     else
%     T_4_energy(i)=PCB_energy(i-4);
%     end
% end
% for i=1: length(PCB_energy)
%     if(i<=5)
%     T_5_energy(i)=PCB_energy(1);
%     else
%     T_5_energy(i)=PCB_energy(i-5);
%     end
% end
% for i=1: length(PCB_energy)
%     if(i<=6)
%     T_6_energy(i)=PCB_energy(1);
%     else
%     T_6_energy(i)=PCB_energy(i-6);
%     end
% end
% for i=1: length(PCB_energy)
%     if(i<=7)
%     T_7_energy(i)=PCB_energy(1);
%     else
%     T_7_energy(i)=PCB_energy(i-7);
%     end
% end
% for i=1: length(PCB_energy)
%     if(i<=8)
%     T_8_energy(i)=PCB_energy(1);
%     else
%     T_8_energy(i)=PCB_energy(i-8);
%     end
% end
% nT_4_energy = normalize(T_4_energy);
% nT_5_energy = normalize(T_5_energy);
% nT_6_energy = normalize(T_6_energy);
% nT_7_energy = normalize(T_7_energy);
% nT_8_energy = normalize(T_8_energy);
 for i=1: length(Tin)
    if(i<=1)
    T_1_Tin(i)=Tin(1);
    else
    T_1_Tin(i)=Tin(i-1);
    end
 end
nT_1_Tin = normalize(T_1_Tin);
 for i=1: length(Tin)
    if(i<=2)
    T_2_Tin(i)=Tin(1);
    else
    T_2_Tin(i)=Tin(i-2);
    end
 end
nT_2_Tin = normalize(T_2_Tin);
 for i=1: length(Tin)
    if(i<=3)
    T_3_Tin(i)=Tin(1);
    else
    T_3_Tin(i)=Tin(i-3);
    end
 end
nT_3_Tin = normalize(T_3_Tin);

%% Flow training
nbStates=20;

% %Flow_data_var = [nOAT;nT_1_flow;nT_2_flow;nT_3_flow;OperationStatus;nFlow];
%Flow_data_var = [nOAT;nRH;nSpaceT;nT_1_flow;nT_2_flow;nT_3_flow;nT_4_flow;nT_5_flow;nT_6_flow;nT_7_flow;nT_8_flow;nDirectR;nDiffuseR;OperationStatus;nFlow];
%Flow_data_var = [nOAT;nRH;nSpaceT;nT_1_flow;nT_2_flow;nT_3_flow;OperationStatus;nFlow];
%Flow_data_var = [nOAT;nRH;nSpaceT;OperationStatus;nFlow];
Flow_data_var = [nPCBControlV;nPress;nFlow];
f_all = size(Flow_data_var,1); %number of input+output
f_var = f_all-1; %number of input
Flow_data_tr = Flow_data_var(:,m:n); %Baselie period
% %Flow_data_tst = Flow_data_var(:,n+1:end);
Flow_data_tst = Flow_data_var(:,tm:tn);
% % 
% % % Training of GMM by EM algorithm, initialized by k-means clustering.
% % [Flow_Priors, Flow_Mu,Flow_Sigma] = EM_init_kmeans(Flow_data_tr, nbStates);
% % [Flow_Priors, Flow_Mu, Flow_Sigma] = EM(Flow_data_tr, Flow_Priors, Flow_Mu, Flow_Sigma);
[Flow_Priors, Flow_Mu,Flow_Sigma] = EM_init_kmeans(Flow_data_tr, nbStates);
[Flow_Priors, Flow_Mu, Flow_Sigma] = EM(Flow_data_tr, Flow_Priors, Flow_Mu, Flow_Sigma);
% expected distribtion 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify input and output variables and train the model using GMR to get
% expected distribution of output variable
Flow_expData1(1:f_var,:)=Flow_data_tr(1:f_var,:);
[Flow_expData1(f_all,:), flow_beta] = GMR(Flow_Priors, Flow_Mu, Flow_Sigma, Flow_expData1(1:f_var,:),[1:f_var],[f_all]);
%sum_flow_beta = sum(flow_beta); %sum of the posterior of flow prediction model

%% Flow prediction
% %Flow_expData(1:f_var,:)=Flow_data_var(1:f_var,:);% all data
Flow_expData(1:f_var,:)=Flow_data_var(1:f_var,m:tn);% all data
[Flow_expData(f_all,:), flow_beta1] = GMR(Flow_Priors, Flow_Mu, Flow_Sigma, Flow_expData(1:f_var,:),[1:f_var],[f_all]);
ntest_flow = Flow_expData(f_all,:); %normalized evolved flow from the evolving gmr model
% 
Predicted_Flow = (Flow_expData(f_all,:))*std(Flow)+mean(Flow);
% 
R_sq_flow_be = 1-sum((Predicted_Flow-Flow(m:tn)).^2)/sum((Predicted_Flow-mean(Flow(m:tn))).^2);
CVrmse_flow_be=sqrt(sum((Predicted_Flow-Flow(m:tn)).^2)/size(Flow(m:tn),2))/mean(Flow(m:tn));
NMBE_flow_be=sum(Predicted_Flow-Flow(m:tn))/size(Flow(m:tn),2)/mean(Flow(m:tn));
% R_sq_flow_be
% CVrmse_flow_be
% NMBE_flow_be
% 
sum_beta_flow=sum(flow_beta1,1);
% 
 %% flow evolving
% [Flow_Priors, Flow_Mu, Flow_Sigma, flow_expData] = Evolving_LW_2(Flow_Priors, Flow_Mu, Flow_Sigma, Flow_data_var(m:tn),sum_beta_flow);
% nevolved_flow = flow_expData.'; %normalixed evolved flow from the evolving gmr model
% evolve_Predicted_Flow = flow_expData*std(Flow)+mean(Flow); %Actual predicted flow after denormalization
% 
% R_sq_flow_ae = 1-sum((evolve_Predicted_Flow-Flow(m:tn)').^2)/sum((evolve_Predicted_Flow-mean(Flow(m:tn))).^2);
% CVrmse_flow_ae=sqrt(sum((evolve_Predicted_Flow-Flow(m:tn)').^2)/size(Flow(m:tn),2))/mean(Flow(m:tn));
% NMBE_flow_ae=sum(evolve_Predicted_Flow-Flow(m:tn)')/size(Flow(m:tn),2)/mean(Flow(m:tn));
% R_sq_flow_ae
% CVrmse_flow_ae
% NMBE_flow_ae
% 
%% Supply temperature training
% nbStates=30;
% 
% % %Flow_data_var = [nOAT;nT_1_flow;nT_2_flow;nT_3_flow;OperationStatus;nFlow];
% %Flow_data_var = [nOAT;nRH;nSpaceT;nT_1_flow;nT_2_flow;nT_3_flow;nT_4_flow;nT_5_flow;nT_6_flow;nT_7_flow;nT_8_flow;nDirectR;nDiffuseR;OperationStatus;nFlow];
% %Flow_data_var = [nOAT;nRH;nSpaceT;nT_1_flow;nT_2_flow;nT_3_flow;OperationStatus;nFlow];
% %Flow_data_var = [nOAT;nRH;nSpaceT;OperationStatus;nFlow];
% Tin_data_var = [nSpaceT;nOAT;nColdValve;nFacadeT;nTin];
% T_all = size(Tin_data_var,1); %number of input+output
% T_var = T_all-1; %number of input
% Tin_data_tr = Tin_data_var(:,m:n); %Baselie period
% %Flow_data_tst = Flow_data_var(:,n+1:end);
% Tin_data_tst = Tin_data_var(:,tm:tn);
% % 
% % % % Training of GMM by EM algorithm, initialized by k-means clustering.
% % [Flow_Priors, Flow_Mu,Flow_Sigma] = EM_init_kmeans(Flow_data_tr, nbStates);
% % [Flow_Priors, Flow_Mu, Flow_Sigma] = EM(Flow_data_tr, Flow_Priors, Flow_Mu, Flow_Sigma);
% [Tin_Priors, Tin_Mu,Tin_Sigma] = EM_init_kmeans(Tin_data_tr, nbStates);
% [Tin_Priors, Tin_Mu, Tin_Sigma] = EM(Tin_data_tr, Tin_Priors, Tin_Mu, Tin_Sigma);
% % expected distribtion 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % specify input and output variables and train the model using GMR to get
% % expected distribution of output variable
% Tin_expData(1:T_var,:)=Tin_data_tr(1:T_var,:);
% [Tin_expData(T_all,:), T_beta] = GMR(Tin_Priors, Tin_Mu, Tin_Sigma, Tin_expData(1:T_var,:),[1:T_var],[T_all]);
% %sum_flow_beta = sum(flow_beta); %sum of the posterior of flow prediction model
% 
% %% Tin prediction
% %Flow_expData(1:f_var,:)=Flow_data_var(1:f_var,:);% all data
% Tin_expData_1(1:T_var,:)=Tin_data_tst(1:T_var,1:tn-tm+1);% all data
% [Tin_expData_1(T_all,:), T_beta1] = GMR(Tin_Priors, Tin_Mu, Tin_Sigma, Tin_expData_1(1:T_var,:),[1:T_var],[T_all]);
% ntest_Tin = Tin_expData_1(T_all,:); %normalized evolved flow from the evolving gmr model
% 
% Predicted_T = (Tin_expData_1(T_all,:))*std(Tin)+mean(Tin);
% 
% R_sq_Tin_be = 1-sum((Predicted_T-Tin(tm:tn)).^2)/sum((Predicted_T-mean(Tin(tm:tn))).^2);
% CVrmse_Tin_be=sqrt(sum((Predicted_T-Tin(tm:tn)).^2)/size(Tin(tm:tn),2))/mean(Tin(tm:tn));
% NMBE_Tin_be=sum(Predicted_T-Tin(tm:tn))/size(Tin(tm:tn),2)/mean(Tin(tm:tn));
% R_sq_Tin_be
% CVrmse_Tin_be
% NMBE_Tin_be
% % 
% sum_beta_Tin=sum(T_beta1,1)
% % 
% %% Tin evolving
% [Tin_Priors, Tin_Mu, Tin_Sigma, Tin_expData_2] = Evolving_LW_2(Tin_Priors, Tin_Mu, Tin_Sigma, Tin_data_var,sum_beta_Tin);
% nevolved_Tin = Tin_expData_2.'; %normalixed evolved flow from the evolving gmr model
% evolve_Predicted_Tin = Tin_expData_2*std(Tin)+mean(Tin); %Actual predicted flow after denormalization
% 
% R_sq_Tin_ae = 1-sum((evolve_Predicted_Tin-Tin').^2)/sum((evolve_Predicted_Tin-mean(Tin)).^2);
% CVrmse_Tin_ae=sqrt(sum((evolve_Predicted_Tin-Tin').^2)/size(Tin,2))/mean(Tin);
% NMBE_Tin_ae=sum(evolve_Predicted_Tin-Tin')/size(Tin,2)/mean(Tin);
% R_sq_Tin_ae
% CVrmse_Tin_ae
% NMBE_Tin_ae

%% Energy data
%Cooling_data_var = [nOAT;OperationStatus;nevolved_flow;nT_1_energy;nT_2_energy;nT_3_energy;nT_4_energy;nT_5_energy;nT_6_energy;nT_7_energy;nT_8_energy;nPCBenergy];
%Cooling_data_var = [nOAT;OperationStatus;nevolved_flow;nT_1_energy;nT_2_energy;nT_3_energy;nT_4_energy;nT_5_energy;nT_6_energy;nT_7_energy;nT_8_energy;nPCBenergy];
%Cooling_data_var =[nOAT;OperationStatus;nevolved_flow;nT_1_energy;nT_2_energy;nT_3_energy;nPCBenergy];%best%0.08
%Cooling_data_var =[nOAT;nRH;nSpaceT;OperationStatus;nevolved_flow;nDirectR;nDiffuseR;nT_1_energy;nT_2_energy;nT_3_energy;nPCBenergy];%best%0.08
%Cooling_data_var =[nOAT;nRH;nSpaceT;OperationStatus;nevolved_flow;nDirectR;nDiffuseR;nT_1_energy;nT_2_energy;nT_3_energy;nPCBenergy];%best%0.08
%Cooling_data_var =[nOAT;nRH;nSpaceT;OperationStatus;ntest_flow;nDirectR;nDiffuseR;nPCBenergy];%best%0.08
%Cooling_data_var =[nOAT(m:tn);nRH(m:tn);nSpaceT(m:tn);nFacadeT(m:tn);nevolved_flow(m:tn);OperationStatus(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%best%0.08
%Cooling_data_var =[nOAT(m:tn);nRH(m:tn);nSpaceT(m:tn);nFacadeT(m:tn);ntest_flow(m:tn);OperationStatus(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%best%0.08
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFlow(m:tn);nTin(m:tn);nT_1_energy;nT_2_energy;nT_3_energy;nPCBenergy(m:tn)];%best%0.08
%Cooling_data_var =[nOAT(m:tn);nRH(m:tn);nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFacadeT(m:tn);nPCBenergy(m:tn)];%S1
%Cooling_data_var =[nOAT(m:tn);nRH(m:tn);nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFacadeT(m:tn);nFlow(m:tn);nPCBenergy(m:tn)];%S1 without flow /S2
%Cooling_data_var =[nOAT(m:tn);nRH(m:tn);nSpaceT(m:tn);nPCBControlV(m:tn);nHotValve(m:tn);nColdValve(m:tn);nFacadeT(m:tn);nFlow(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%S3
%Cooling_data_var =[nOAT(m:tn);nRH(m:tn);nSpaceT(m:tn);nPCBControlV(m:tn);nHotValve(m:tn);nColdValve(m:tn);nFacadeT(m:tn);nFlow(m:tn);nPCBenergy(m:tn)];%S3

%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFacadeT(m:tn);nFlow(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%S3
%Cooling_data_var=[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFlow(m:tn);nTin(m:tn);nT_1_energy;nPCBenergy(m:tn)];%best%0.6063
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nTin(m:tn);nT_1_energy;nT_2_energy;nT_3_energy;nPCBenergy(m:tn)];%best%0.4892
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);ntest_Tin(m:tn);nT_1_energy;nT_2_energy;nT_3_energy;nPCBenergy(m:tn)];%best%0.4892
%scatter(SpaceT(m:tn),PCBenergy(m:tn));  
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFlow(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%best%0.2327
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFlow(m:tn);nPCBenergy(m:tn)];%best%0.2428
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%best%0.2428
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);ntest_flow;nT_1_energy(m:tn);nPCBenergy(m:tn)];%best%0.2428
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFlow(m:tn);nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%best%0.489
%Cooling_data_var =[nSpaceT(m:tn);nPCBControlV(m:tn);nColdValve(m:tn);nFlow(m:tn);ntest_Tin;nT_1_energy(m:tn);nT_2_energy(m:tn);nT_3_energy(m:tn);nPCBenergy(m:tn)];%best%0.489

%Cooling_data_var = [nSWT;nOAT;SpaceT;nevolved_flow;V1;V2;V3;nT_1_energy;nT_2_energy;OperationStatus_Pump;nPCBenergy];

%Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFlow(m:n);nT_1_energy(m:n);nT_2_energy(m:n);nT_3_energy(m:n);nPCBenergy(m:n)];%best%0.489 ; %Baseline training data
%Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFlow(m:n);nPCBenergy(m:n)];%best%0.489 ; %Baseline training data
%Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);ntest_flow(1:n-m+1);nT_1_energy(m:n);nT_2_energy(m:n);nT_3_energy(m:n);nPCBenergy(m:n)];%best%0.489 ; %Baseline training data
%Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nHotValve(m:n);nOAT(m:n);nRH(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nT_1_energy(m:n);nT_2_energy(m:n);nT_3_energy(m:n);nPCBenergy(m:n)];%best%0.489 ; %Baseline training data
%Cooling_data_tr =[nOAT(m:n);nRH(m:n);nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFacadeT(m:n);nPCBenergy(m:n)];%S1
%Cooling_data_tr =[nOAT(m:n);nRH(m:n);nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nPCBenergy(m:n)];%S1 without flow; S2 with flow
%Cooling_data_tr =[nOAT(m:n);nRH(m:n);nSpaceT(m:n);nPCBControlV(m:n);nHotValve(m:n);nColdValve(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nT_1_energy(m:n);nT_2_energy(m:n);nT_3_energy(m:n);nPCBenergy(m:n)];%S3
%Cooling_data_tr =[nOAT(m:n);nRH(m:n);nSpaceT(m:n);nPCBControlV(m:n);nHotValve(m:n);nColdValve(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nPCBenergy(m:n)];%S3
%Cooling_data_tr =[nOAT(m:n);nRH(m:n);nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nPCBenergy(m:n)];%S3
Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nPCBenergy(m:n)];%S3

%Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);nFacadeT(m:n);ntest_flow(1:n-m+1);nT_1_energy(m:n);nT_2_energy(m:n);nT_3_energy(m:n);nPCBenergy(m:n)];%S3

%Cooling_data_tr =[nSpaceT(m:n);nPCBControlV(m:n);nColdValve(m:n);ntest_flow(1:n-m+1);nT_1_energy(m:n);nPCBenergy(m:n)];%best%0.489 ; %Baseline training data
%Cooling_data_tst = Cooling_data_var(:,n+1:end);
%Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nFlow(tm:tn);nT_1_energy(tm:tn);nT_2_energy(tm:tn);nT_3_energy(tm:tn);nPCBenergy(tm:tn)];%best%0.489
%Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nFlow(tm:tn);nPCBenergy(tm:tn)];%best%0.489
%Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);ntest_flow(tm-m+1:tn-m+1);nT_1_energy(tm:tn);nT_2_energy(tm:tn);nT_3_energy(tm:tn);nPCBenergy(tm:tn)];%best%0.489
%Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nHotValve(tm:tn);nOAT(tm:tn);nRH(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nT_1_energy(tm:tn);nT_2_energy(tm:tn);nT_3_energy(tm:tn);nPCBenergy(tm:tn)];%best%0.489
%Cooling_data_tst = [nOAT(tm:tn);nRH(tm:tn);nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nPCBenergy(tm:tn)];%S1 without flow; S2 with flow
%Cooling_data_tst = [nOAT(tm:tn);nRH(tm:tn);nSpaceT(tm:tn);nPCBControlV(tm:tn);nHotValve(tm:tn);nColdValve(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nT_1_energy(tm:tn);nT_2_energy(tm:tn);nT_3_energy(tm:tn);nPCBenergy(tm:tn)];%S3
%Cooling_data_tst = [nOAT(tm:tn);nRH(tm:tn);nSpaceT(tm:tn);nPCBControlV(tm:tn);nHotValve(tm:tn);nColdValve(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nPCBenergy(tm:tn)];%S3
%Cooling_data_tst = [nOAT(tm:tn);nRH(tm:tn);nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nPCBenergy(tm:tn)];%S3
Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nPCBenergy(tm:tn)];%S3
%Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);nFacadeT(tm:tn);ntest_flow(tm-m+1:tn-m+1);nT_1_energy(tm:tn);nT_2_energy(tm:tn);nT_3_energy(tm:tn);nPCBenergy(tm:tn)];%S3
%Cooling_data_tst = [nSpaceT(tm:tn);nPCBControlV(tm:tn);nColdValve(tm:tn);ntest_flow(tm-m+1:tn-m+1);nT_1_energy(tm:tn);nPCBenergy(tm:tn)];%best%0.489

Measured_energy = PCBenergy(:,m:tn).';
scatter(SpaceT(m:tn),PCBenergy(m:tn));    
%% train PCB_Energy
scvrmse=[];
for i=30
%nbStates =i;%30;best scenario
nbStates=i;
nall = size(Cooling_data_tr,1); %number of input+output
nvar = nall-1; %number of input
% % Training of GMM by EM algorithm, initialized by k-means clustering.
[ePriors, eMu,eSigma] = EM_init_kmeans(Cooling_data_tr, nbStates);
[ePriors, eMu, eSigma] = EM(Cooling_data_tr, ePriors, eMu, eSigma);

% expected distribtion 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify input and output variables and train the model using GMR to get
% expected distribution of output variable
Energy_train_expData(1:nvar,:)=Cooling_data_tr(1:nvar,:);
[Energy_train_expData(nall,:), beta_energy] = GMR(ePriors, eMu, eSigma, Energy_train_expData(1:nvar,:),[1:nvar],[nall]);
Predicted_cooling_energy = (Energy_train_expData(nall,:).')*std(PCBenergy)+mean(PCBenergy);
Measured_Predicted = [PCBenergy(m:n).' Predicted_cooling_energy];% measured and predicted cooling energy during baseline period
for j = 1:2
    for i = 1:size(Measured_Predicted,1)
        if Measured_Predicted(i,j)<0
        Measured_Predicted(i,j)=[0];
        end
    end
end
sum_beta_energy = sum(beta_energy,1); %sum of the posterior of energy prediction model
%% testing  PCB_Energy before evolving for 
Energy_test_expData2(1:nvar,:)=Cooling_data_tst(1:nvar,:);
[Energy_test_expData2(nall,:), beta2]= GMR(ePriors, eMu, eSigma, Energy_test_expData2(1:nvar,:),[1:nvar],[nall]);

Predicted_cooling_energy_test2 = (Energy_test_expData2(nall,:).')*std(PCBenergy)+mean(PCBenergy);

for j = 1
    for i = 1:size(Predicted_cooling_energy_test2)
        if (Predicted_cooling_energy_test2(i,j)<0)|(SpaceT(j:i)>Tin(j:i))
        Predicted_cooling_energy_test2(i,j)=[0];
        end
    end
end

%energy_mdl_beforeEvolving=fitlm(Measured_energy,Predicted_cooling_energy_test2);
%R_sq_energy_be = energy_mdl_beforeEvolving.Rsquared.Ordinary;
R_sq_energy_be = 1-sum((Predicted_cooling_energy_test2-PCBenergy(:,tm:tn)').^2)/sum((Predicted_cooling_energy_test2-mean(PCBenergy(:,tm:tn))).^2);
%CVrmse_energy_be = energy_mdl_beforeEvolving.RMSE/mean(Measured_energy);
CVrmse_energy_be=sqrt(sum((Predicted_cooling_energy_test2-PCBenergy(:,tm:tn)').^2)/length(PCBenergy(:,tm:tn)))/mean(PCBenergy(:,tm:tn));
NMBE_energy_be=sum(Predicted_cooling_energy_test2-PCBenergy(:,tm:tn)')/length(PCBenergy(:,tm:tn))/mean(PCBenergy(:,tm:tn));
R_sq_energy_be
CVrmse_energy_be
NMBE_energy_be
%% Energy evolving
[ePriors, eMu, eSigma, energy_expData] = Evolving_LW_2(ePriors, eMu, eSigma, Cooling_data_tst, sum_beta_energy);
%[ePriors, eMu, eSigma, energy_expData] = Evolving_new(ePriors, eMu, eSigma, Cooling_data_tst,0.01, 5, 0.0001, 10, sum_beta);

Predicted_cooling_energy_evolve = energy_expData*std(PCBenergy')+mean(PCBenergy'); %Actual predicted energy after denormalization

for j = 1
    for i = 1:size(Predicted_cooling_energy_evolve)
        if (Predicted_cooling_energy_evolve(i,j)<0)|(SpaceT(j:i)>Tin(j:i))
        Predicted_cooling_energy_evolve(i,j)=[0];
        end
    end
end

%energy_mdl_afterEvolving=fitlm(Measured_energy,Predicted_cooling_energy_evolve);
%R_sq_energy_ae = energy_mdl_afterEvolving.Rsquared.Ordinary;
%CVrmse_energy_ae = energy_mdl_afterEvolving.RMSE/mean(Measured_energy);
%save 'PCB_parameters_correl' ePriors eMu eSigma sum_beta Flow_Priors Flow_Mu Flow_Sigma sum_flow_beta;
%CVRMSE_Rsq_all = [CVrmse_flow_be CVrmse_flow_ae CVrmse_energy_be CVrmse_energy_ae;R_sq_flow_be R_sq_flow_ae R_sq_energy_be R_sq_energy_ae];
R_sq_energy_ae = 1- sum((Predicted_cooling_energy_evolve-PCBenergy(:,tm:tn)').^2)/sum((Predicted_cooling_energy_evolve-mean(PCBenergy(:,tm:tn)')).^2);
CVrmse_energy_ae=sqrt(sum((Predicted_cooling_energy_evolve-PCBenergy(:,tm:tn)').^2)/length(PCBenergy(:,tm:tn)'))/mean(PCBenergy(:,tm:tn)');
NMBE_energy_ae=sum(Predicted_cooling_energy_evolve-PCBenergy(:,tm:tn)')/length(PCBenergy(:,tm:tn)')/mean(PCBenergy(:,tm:tn)');
R_sq_energy_ae
CVrmse_energy_ae
NMBE_energy_ae

scvrmse=[scvrmse; R_sq_energy_be CVrmse_energy_be NMBE_energy_be R_sq_energy_ae CVrmse_energy_ae NMBE_energy_ae]
end

% CVRMSE_Rsq_all = [CVrmse_flow_be CVrmse_flow_ae;NMBE_flow_be NMBE_flow_ae;R_sq_flow_be R_sq_flow_ae;CVrmse_energy_be CVrmse_energy_ae;NMBE_energy_be NMBE_energy_ae;R_sq_energy_be R_sq_energy_ae];

%% testing  PCB_Energy for faults

%% Load a dataset
fm=1; %first data point for training (April 1st)
fn=1272; %3617; %end data point for training baseline
B = csvread('TT_out_2021_Fault.csv',2,1);
fdata_var = B.'
fdata_tst = B(fm:fn,:); %Baselsie period  Janurary
%% Asssigining variables
fOAT=fdata_var(9,:);
nfOAT=(fOAT-mean(OAT))/std(OAT);
fSpaceT=(fdata_var(11,:));
nfSpaceT=(fSpaceT-mean(SpaceT))/std(SpaceT);
fRH=fdata_var(12,:);
nfRH=(fRH-mean(RH))/std(RH);
fFacadeT=fdata_var(13,:);
nfFacadeT=(fFacadeT-mean(FacadeT))/std(FacadeT);
fPCBControlV=(fdata_var(6,:)+fdata_var(7,:)+fdata_var(8,:))/3;
nfPCBControlV=(fPCBControlV-mean(PCBControlV))/std(PCBControlV);
%SpaceH=data_var(2,:);
%nSpaceH=normalize(SpaceH);
fHotValve=fdata_var(4,:);
nfHotValve=(fHotValve-mean(HotValve))/std(HotValve);
fColdValve=fdata_var(5,:);
nfColdValve=(fColdValve-mean(ColdValve))/std(ColdValve);
%Pump_speed=data_var(19,:);
%nPump_speed = normalize(Pump_speed);
fFlow=fdata_var(16,:);
nfFlow=(fFlow-mean(Flow))/std(Flow);
fPress=fdata_var(3,:);
nfPress = (fPress-mean(Press))/std(Press);
fTin=fdata_var(14,:);% supply water temperature
nfTin=normalize(Tin);
% AHU_SAT=data_var(10,:);
% nAHU_SAT=normalize(AHU_SAT);
%CHWST=data_var(9,:);
%nCHWST=normalize(CHWST);
%% Flow prediction fault data
% %Flow_expData(1:f_var,:)=Flow_data_var(1:f_var,:);% all data
f_Flow_data_var = [nfPCBControlV;nfPress];
f_Flow_expData(1:f_var,:)=f_Flow_data_var(1:f_var,fm:fn);% all data
[f_Flow_expData(f_all,:), f_flow_beta1] = GMR(Flow_Priors, Flow_Mu, Flow_Sigma, f_Flow_expData(1:f_var,:),[1:f_var],[f_all]);
nftest_flow = f_Flow_expData(f_all,:); %normalized evolved flow from the evolving gmr model
f_Predicted_Flow = (f_Flow_expData(f_all,:))*std(Flow)+mean(Flow);
%% Energy prediciton fault data
%f_Cooling_data_var =[nfOAT(fm:fn);nfRH(fm:fn);nfSpaceT(fm:fn);nfPCBControlV(fm:fn);nfHotValve(fm:fn);nfColdValve(fm:fn);nfFacadeT(fm:fn);nftest_flow(fm:fn)];%S3
f_Cooling_data_var =[nfSpaceT(fm:fn);nfPCBControlV(fm:fn);nfColdValve(fm:fn);nfFacadeT(fm:fn);nftest_flow(fm:fn)];%S3
f_Energy_test_expData2(1:nvar,:)=f_Cooling_data_var(1:nvar,:);
[f_Energy_test_expData2(nall,:), beta2]= GMR(ePriors, eMu, eSigma, f_Energy_test_expData2(1:nvar,:),[1:nvar],[nall]);
f_Predicted_cooling_energy = (f_Energy_test_expData2(nall,:).')*std(PCBenergy)+mean(PCBenergy);

for j = 1
    for i = 1:size(f_Predicted_cooling_energy)
        if (f_Predicted_cooling_energy(i,j)<0)|(fSpaceT(j:i)>fTin(j:i))
        f_Predicted_cooling_energy(i,j)=[0];
        end
    end
end
% Energy_test_expData3(1:nvar,:)=data_fault_test(1:nvar,:);
% 
% [Energy_test_expData3(nall,:), beta3]= GMR(ePriors, eMu, eSigma, Energy_test_expData3(1:nvar,:),[1:nvar],[nall]);
% 
% Predicted_cooling_energy_test3 = (Energy_test_expData3(nall,:).')*std(PCB_energy)+mean(PCB_energy);
% 
% for m = 1
%     for i = 1:size(Predicted_cooling_energy_test3)
%         if Predicted_cooling_energy_test3(i,m)<0
%         Predicted_cooling_energy_test3(i,m)=[0];
%         end
%     end
% end
