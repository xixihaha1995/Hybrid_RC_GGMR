function gmr_rs
%% Definition of the number of components used in GMM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbStates = 4;

%% Convert RC training data to GMR training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfile('data/case_arr.mat')
    T=readtable('data/case_arr.csv');
    t_out=T{:,1};
    t_slabs= T{:,6};
    t_cav = T{:,50};
    t_water_sup = (T{:,29}-32)*5/9;
    t_water_ret = (T{:,30} - 32) *5/9;
    vfr_water = T{:,28};
    q_solar = T{:,75};
    q_light = T{:,81};
    q_inte_heat = T{:,79};
    ahu_cfm1 = T{:,42};
    ahu_t_sup1 = T{:,43};
    ahu_cfm2 = T{:,46};
    ahu_t_sup2 = T{:,47};
    c_water = 4.186;
    rho_water = 997e3;
    gal_per_min_to_m3 = 6.309e-5;
    y= c_water*rho_water*gal_per_min_to_m3*vfr_water.*(t_water_sup - t_water_ret);
    data = [t_slabs,t_cav,t_water_sup,t_water_ret,vfr_water,q_solar,...
        q_light,q_inte_heat,ahu_cfm1,ahu_t_sup1,ahu_cfm2,ahu_t_sup2,t_out,y];
    data = data.';
    save('data/case_arr.mat','data');
end
%% Load a dataset consisting of 3 demonstrations of a 2D signal.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/case_arr.mat'); %load 'Data'
Data = data(:,1:100);
testData = data(:,100:200);

nbVar = size(Data,1);

%% Training of GMM by EM algorithm, initialized by k-means clustering.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);

%% Use of GMR to retrieve a generalized version of the data and associated
%% constraints. A sequence of temporal values is used as input, and the 
%% expected distribution is retrieved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:13
    expData(i,:) = linspace(min(Data(i,:)), max(Data(i,:)), 100);
end
[expData(14,:), expSigma] = GMR(Priors, Mu, Sigma,  expData(1,:), [1:13], [14]);

%% Plot of the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1000,800],'name','GMM-GMR-rs');
% %plot 1D
% 
% for n=1:nbVar-1
%   subplot(3*(nbVar-1),2,(n-1)*2+1); hold on;
%   plot(Data(1,:), Data(n+1,:), 'x', 'markerSize', 4, 'color', [.3 .3 .3]);
%   axis([min(Data(1,:)) max(Data(1,:)) min(Data(n+1,:))-0.01 max(Data(n+1,:))+0.01]);
%   xlabel('t','fontsize',16); ylabel(['x_' num2str(n)],'fontsize',16);
% end
% %plot 2D
% subplot(3*(nbVar-1),2,[2:2:2*(nbVar-1)]); hold on;
% plot(Data(2,:), Data(3,:), 'x', 'markerSize', 4, 'color', [.3 .3 .3]);
% axis([min(Data(2,:))-0.01 max(Data(2,:))+0.01 min(Data(3,:))-0.01 max(Data(3,:))+0.01]);
% xlabel('x_1','fontsize',16); ylabel('x_2','fontsize',16);
% 
% 
% nbVar = old_nbVar;
% % %% Plot of the GMM encoding results
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %plot 1D
% for n=1:nbVar-1
%   subplot(3*(nbVar-1),2,4+(n-1)*2+1); hold on;
%   plotGMM(Mu([1,n+1],:), Sigma([1,n+1],[1,n+1],:), [0 .8 0], 1);
%   axis([min(Data(1,:)) max(Data(1,:)) min(Data(n+1,:))-0.01 max(Data(n+1,:))+0.01]);
%   xlabel('t','fontsize',16); ylabel(['x_' num2str(n)],'fontsize',16);
% end
% %plot 2D
% subplot(3*(nbVar-1),2,4+[2:2:2*(nbVar-1)]); hold on;
% plotGMM(Mu([2,3],:), Sigma([2,3],[2,3],:), [0 .8 0], 1);
% axis([min(Data(2,:))-0.01 max(Data(2,:))+0.01 min(Data(3,:))-0.01 max(Data(3,:))+0.01]);
% xlabel('x_1','fontsize',16); ylabel('x_2','fontsize',16);
% 
% %% Plot of the GMR regression results
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %plot 1D
% for n=1:nbVar-1
%   subplot(3*(nbVar-1),2,8+(n-1)*2+1); hold on;
%   plotGMM(expData([1,n+1],:), expSigma(n,n,:), [0 0 .8], 3);
%   axis([min(Data(1,:)) max(Data(1,:)) min(Data(n+1,:))-0.01 max(Data(n+1,:))+0.01]);
%   xlabel('t','fontsize',16); ylabel(['x_' num2str(n)],'fontsize',16);
% end
% %plot 2D
% subplot(3*(nbVar-1),2,8+[2:2:2*(nbVar-1)]); hold on;
% plotGMM(expData([2,3],:), expSigma([1,2],[1,2],:), [0 0 .8], 2);
% axis([min(Data(2,:))-0.01 max(Data(2,:))+0.01 min(Data(3,:))-0.01 max(Data(3,:))+0.01]);
% xlabel('x_1','fontsize',16); ylabel('x_2','fontsize',16);
% 
% % pause;
% close all;
