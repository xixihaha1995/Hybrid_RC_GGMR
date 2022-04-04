function gmr_rs
%% Definition of the number of components used in GMM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
nbStates = 14;

%% Convert RC training data to GMR training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfile('data/case_arr_sim.mat')
    T=readtable('data/case_arr.csv');
    t_out=(T{:,1} - 32) *5/9;
    t_slabs= (T{:,6} - 32 ) * 5/9;
    t_cav = (T{:,50} - 32) * 5/ 9;
    t_water_sup = (T{:,29}-32)*5/9;
    t_water_ret = (T{:,30} - 32) *5/9;
    vfr_water = T{:,28};
    m3_per3_perCFM = 0.00047194745;
    ahu_cfm1 = T{:,42} *  m3_per3_perCFM;
    ahu_t_sup1 = (T{:,43}  - 32) * 5/9;
    ahu_cfm2 = T{:,46} * m3_per3_perCFM;
    ahu_t_sup2 = (T{:,47} - 32 ) *5/9;
    
%     q_solar = T{:,75};
%     q_light = T{:,81};
%     q_inte_heat = T{:,79};
    
    c_water = 4.186;
    rho_water = 997e3;
    gal_per_min_to_m3 = 6.309e-5;
    y= c_water*rho_water*gal_per_min_to_m3*vfr_water.*(t_water_sup - t_water_ret);
    
    data = [t_slabs,t_cav,t_water_sup,t_water_ret,vfr_water,...
        ahu_cfm1,ahu_t_sup1,ahu_cfm2,ahu_t_sup2,t_out,y];
    data = data.';
    save('data/case_arr_sim.mat','data');
end

if ~isfile('data/case_arr.mat')
    T=readtable('data/case_arr.csv');
    t_out=(T{:,1} - 32) *5/9;
    t_slabs= (T{:,6} - 32 ) * 5/9;
    t_cav = (T{:,50} - 32) * 5/ 9;
    t_water_sup = (T{:,29}-32)*5/9;
    t_water_ret = (T{:,30} - 32) *5/9;
    vfr_water = T{:,28};
    m3_per3_perCFM = 0.00047194745;
    ahu_cfm1 = T{:,42} *  m3_per3_perCFM;
    ahu_t_sup1 = (T{:,43}  - 32) * 5/9;
    ahu_cfm2 = T{:,46} * m3_per3_perCFM;
    ahu_t_sup2 = (T{:,47} - 32 ) *5/9;
    
    q_solar = T{:,75};
    q_light = T{:,81};
    q_inte_heat = T{:,79};
    
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
load('data/case_arr_sim.mat'); %load 'Data'
total_length = size(data,2);
training_length = 4032;
testing_length = total_length - training_length;
nbVarAll = size(data,1);
nbVarInput = nbVarAll - 1;

% train_no_norm = data(:,1:training_length);
% test_no_norm = data(:,training_length+1 :training_length+testing_length);
% training_data = normalize(train_no_norm);
% testing_data = normalize(test_no_norm);

train_input_no_norm = data(1:nbVarInput,1:training_length);
training_input_data=normalize(train_input_no_norm);
training_data= [training_input_data; data(nbVarAll,1:training_length)];

test_input_no_norm = data(1:nbVarInput,training_length+1:training_length+testing_length);
testing_input_data=normalize(test_input_no_norm);
testing_data= [testing_input_data; data(nbVarAll,training_length+1:training_length+testing_length)];
%% Training of GMM by EM algorithm, initialized by k-means clustering.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Priors, Mu, Sigma] = EM_init_kmeans(training_data, nbStates);
[Priors, Mu, Sigma] = EM(training_data, Priors, Mu, Sigma);

%% Use of GMR to retrieve a generalized version of the data and associated
%% constraints. A sequence of temporal values is used as input, and the 
%% expected distribution is retrieved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[expData(nbVarAll,:), expSigma] = GMR(Priors, Mu, Sigma,  testing_data(1:nbVarInput,:), [1:nbVarInput], [nbVarAll]);

%% Plot of the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,500],'name','GMM-GMR-rs');
subplot(1,1,1); hold on;

reverse_norm_model_test_y = expData(nbVarAll,:);
actual_test_y = testing_data(nbVarAll,:);
% train_y_mean = mean(train_no_norm(nbVarAll,:));
% train_y_std = std(train_no_norm(nbVarAll,:));
% actual_test_y = test_no_norm(nbVarAll,:);
% reverse_norm_model_test_y = expData(nbVarAll,:) * train_y_std + train_y_mean;

xlabel('Time step, 5 min interval') 
ylabel('Radiant Slab Loads (W)') 
plot(reverse_norm_model_test_y);
plot(actual_test_y);

rmse = (sum((reverse_norm_model_test_y - actual_test_y).^2) / length(actual_test_y)).^ (0.5); 
mean_model = mean(abs(reverse_norm_model_test_y));
std_model = (sum((reverse_norm_model_test_y - mean_model).^2) / length(reverse_norm_model_test_y)) .^ (0.5); 
nrmse = rmse *100 / std_model;

mean_measured = mean(abs(actual_test_y));
cvrmse = rmse*100 / mean_measured;

mae = sum(abs(actual_test_y - reverse_norm_model_test_y)) / length(actual_test_y);

mape_ratio = abs(actual_test_y - reverse_norm_model_test_y) ./ abs(actual_test_y);
mape_ratio(isinf(mape_ratio)) = 0;
mape = sum(mape_ratio)*100 / length(actual_test_y);
    
title({"NRMSE is " + nrmse + "%","CVRMSE is " + cvrmse + "%",...
    "MAE is " + mae + "W","MAPE is " + mape + "%"})
legend({'Predicted','Actual'},'Location','southwest');
