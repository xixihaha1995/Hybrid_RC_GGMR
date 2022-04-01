function gmr_rs
%% Definition of the number of components used in GMM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbStates = 10;

%% Convert RC training data to GMR training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfile('data/case_arr.mat')
    T=readtable('data/case_arr.csv');
    t_out=(T{:,1} - 32) *5/9;
    t_slabs= (T{:,6} - 32 ) * 5/9;
    t_cav = (T{:,50} - 32) * 5/ 9;
    t_water_sup = (T{:,29}-32)*5/9;
    t_water_ret = (T{:,30} - 32) *5/9;
    vfr_water = T{:,28};
    q_solar = T{:,75};
    q_light = T{:,81};
    q_inte_heat = T{:,79};
    m3_per3_perCFM = 0.00047194745;
    ahu_cfm1 = T{:,42} *  m3_per3_perCFM;
    ahu_t_sup1 = (T{:,43}  - 32) * 5/9;
    ahu_cfm2 = T{:,46} * m3_per3_perCFM;
    ahu_t_sup2 = (T{:,47} - 32 ) *5/9;
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
total_length = size(data,2);
training_length = 1000;
testing_length = 100;

train_no_norm = data(:,1:training_length);
test_no_norm = data(:,training_length+1 :training_length+testing_length);

training_data = normalize(train_no_norm);
testing_data = normalize(test_no_norm);

nbVarAll = size(data,1);
nbVarInput = nbVarAll - 1;
%% Training of GMM by EM algorithm, initialized by k-means clustering.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Priors, Mu, Sigma] = EM_init_kmeans(training_data, nbStates);
[Priors, Mu, Sigma] = EM(training_data, Priors, Mu, Sigma);

%% Use of GMR to retrieve a generalized version of the data and associated
%% constraints. A sequence of temporal values is used as input, and the 
%% expected distribution is retrieved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nbVarInput
    expData(i,:) = linspace(min(testing_data(i,:)), max(testing_data(i,:)), testing_length);
end
[expData(nbVarAll,:), expSigma] = GMR(Priors, Mu, Sigma,  expData(1,:), [1:nbVarInput], [nbVarAll]);

%% Plot of the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1000,800],'name','GMM-GMR-rs');
subplot(1,1,1); hold on;
train_y_mean = mean(train_no_norm(nbVarAll,:));
train_y_std = std(train_no_norm(nbVarAll,:));
reverse_norm_model_test_y = expData(nbVarAll,:) * train_y_std + train_y_mean;
actual_test_y = test_no_norm(nbVarAll,:);
xlabel('Time step, 5 min interval') 
ylabel('Radiant Slab Loads (W)') 
plot(reverse_norm_model_test_y);
plot(actual_test_y);
legend({'Predicted','Actual'},'Location','southwest')
close all;
