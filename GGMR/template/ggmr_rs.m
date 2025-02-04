clear
%% Convert RC training data to GMR/GGMR training data
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
    
    q_solar = T{:,75};
    q_light = T{:,81};
    q_inte_heat = T{:,79};
    
    c_water = 4.186;
    rho_water = 997e3;
    gal_per_min_to_m3 = 6.309e-5;
    y= c_water*rho_water*gal_per_min_to_m3*vfr_water.*(t_water_sup - t_water_ret);
    
    t_slabs = t_slabs.';
    t_cav = t_cav.';
    t_water_sup = t_water_sup.';
    t_water_ret = t_water_ret.';
    vfr_water = vfr_water.';
    q_solar = q_solar.';
    q_light = q_light.';
    q_inte_heat = q_inte_heat.';
    ahu_cfm1 = ahu_cfm1.';
    ahu_cfm2 = ahu_cfm2.';
    ahu_t_sup2 = ahu_t_sup2.';
    t_out = t_out.';
    y = y.';
    save('data/case_arr_sim.mat','t_slabs','t_cav','t_water_sup','t_water_ret','vfr_water','q_solar',...
        'q_light','q_inte_heat','ahu_cfm1','ahu_t_sup1','ahu_cfm2','ahu_t_sup2','t_out','y');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Flow training
load('data/case_arr_sim.mat'); %load 'Data'
t_out_norm = normalize(t_out);
t_slabs_norm = normalize(t_slabs);
t_cav_norm = normalize(t_cav);
t_water_sup_norm = normalize(t_water_sup);
t_water_ret_norm = normalize(t_water_ret);
vfr_water_norm = normalize(vfr_water);
y_norm = normalize(y);

rs_data_var_norm_all = [t_out_norm; t_slabs_norm; t_cav_norm; t_water_sup_norm;...
    t_water_ret_norm; vfr_water_norm; y_norm];

total_length = size(rs_data_var_norm_all,2);
training_length = 4032;
testing_length = total_length - training_length;
nbVarAll = size(rs_data_var_norm_all,1);
nbVarInput = nbVarAll - 1;

rs_data_var_norm_train = rs_data_var_norm_all(:,1:training_length);
rs_data_var_norm_test = rs_data_var_norm_all(:,training_length+1 :training_length+testing_length);
y_train = y(:,1:training_length);
y_test = y(:,training_length+1 :training_length+testing_length);

nbStates=20;
%% Flow prediction using slightly modified GMR
[rs_Priors, rs_Mu, rs_Sigma] = EM_init_kmeans(rs_data_var_norm_train, nbStates);
[rs_Priors, rs_Mu, rs_Sigma]  = EM(rs_data_var_norm_train, rs_Priors, rs_Mu, rs_Sigma);

[rs_expData_gmr_norm, rs_beta] = GMR(rs_Priors, rs_Mu, rs_Sigma, rs_data_var_norm_test(1:nbVarInput,:),[1:nbVarInput],[nbVarAll]);
rs_expData_gmr = rs_expData_gmr_norm * std(y_train)+ mean(y_train);

rmse_gmr = (sum((rs_expData_gmr - y_test).^2) / length(y_test)).^ (0.5); 
mean_measured_gmr = mean(abs(y_test));
cvrmse_gmr = rmse_gmr*100 / mean_measured_gmr;
%% Flow prediction using Evolving GMR /GGMR
sum_beta_rs=sum(rs_beta,1);
[rs_Priors, rs_Mu, rs_Sigma, rs_expData_ggmr_norm] = Evolving_LW_2(rs_Priors, rs_Mu, rs_Sigma, rs_data_var_norm_test,sum_beta_rs);
rs_expData_ggmr_norm = rs_expData_ggmr_norm.';
rs_expData_ggmr= rs_expData_ggmr_norm*std(y_train)+mean(y_train); %Actual predicted flow after denormalization

rmse_ggmr = (sum((rs_expData_ggmr - y_test).^2) / length(y_test)).^ (0.5); 
mean_measured_ggmr = mean(abs(y_test));
cvrmse_ggmr = rmse_ggmr*100 / mean_measured_ggmr;

%% Plot
figure('position',[10,10,1000,800],'name','GMR-GGMR-RS-Load Prediction');
subplot(1,1,1); hold on;

xlabel('Time step, 5 min interval') 
ylabel('Radiant Slab Loads (W)') 
plot(rs_expData_gmr);
plot(rs_expData_ggmr);
plot(actual_test_y);
legend({'GMR Predicted','GGMR Predicted','Actual'},'Location','southwest')
title("CVRMSE, GMR:" + cvrmse_gmr + "%, GGMR:"+ cvrmse_ggmr + "%")