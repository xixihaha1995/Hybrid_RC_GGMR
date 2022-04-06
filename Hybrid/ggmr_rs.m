function [cvrmse_gmr, cvrmse_ggmr] = ggmr_rs(nbStates, input_case)
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
    rc_y = T{:,82};
    
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
    ahu_t_sup1 = ahu_t_sup1.';
    ahu_t_sup2 = ahu_t_sup2.';
    t_out = t_out.';
    rc_y = rc_y.';
    y = y.';
    save('data/case_arr_sim.mat','t_slabs','t_cav','t_water_sup','t_water_ret','vfr_water','q_solar',...
        'q_light','q_inte_heat','ahu_cfm1','ahu_t_sup1','ahu_cfm2','ahu_t_sup2','t_out','rc_y','y');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Flow training
load('data/case_arr_sim.mat'); %load 'Data'
t_out_norm = normalize(t_out);
t_slabs_norm = normalize(t_slabs);
t_cav_norm = normalize(t_cav);
% t_water_sup_norm = normalize(t_water_sup);
% t_water_ret_norm = normalize(t_water_ret);
% vfr_water_norm = normalize(vfr_water);
ahu_cfm1_norm = normalize(ahu_cfm1);
ahu_cfm2_norm = normalize(ahu_cfm2);
ahu_t_sup1_norm = normalize(ahu_t_sup1);
ahu_t_sup2_norm = normalize(ahu_t_sup2);
q_solar_norm = normalize(q_solar);
q_light_norm = normalize(q_light);
q_inte_heat_norm = normalize(q_inte_heat);
rc_y_norm = normalize(rc_y);
y_norm = normalize(y);


switch (input_case)
    case 1
        rs_data_var_norm_all = [t_out_norm; t_slabs_norm; t_cav_norm; rc_y_norm; y_norm];
    case 2
        rs_data_var_norm_all = [t_out_norm; t_slabs_norm; t_cav_norm;...
            ahu_cfm1_norm; ahu_t_sup1_norm; ahu_cfm2_norm; ahu_t_sup2_norm; y_norm];
    case 3
        rs_data_var_norm_all = [t_out_norm; t_slabs_norm; t_cav_norm;...
            q_solar_norm; q_light_norm; q_inte_heat_norm; y_norm];
    case 4
        rs_data_var_norm_all = [t_out_norm; t_slabs_norm; t_cav_norm;...
            ahu_cfm1_norm; ahu_t_sup1_norm; ahu_cfm2_norm; ahu_t_sup2_norm;...
            q_solar_norm; q_light_norm; q_inte_heat_norm;rc_y_norm; y_norm];
end

total_length = size(rs_data_var_norm_all,2);
training_length = 4032;
testing_length = total_length - training_length;
nbVarAll = size(rs_data_var_norm_all,1);
nbVarInput = nbVarAll - 1;

rs_data_var_norm_train = rs_data_var_norm_all(:,1:training_length);
rs_data_var_norm_test = rs_data_var_norm_all(:,training_length+1 :training_length+testing_length);
y_train = y(:,1:training_length);
y_test = y(:,training_length+1 :training_length+testing_length);


%% RS Load prediction using GMR
[rs_Priors, rs_Mu, rs_Sigma] = EM_init_kmeans(rs_data_var_norm_train, nbStates);
[rs_Priors, rs_Mu, rs_Sigma]  = EM(rs_data_var_norm_train, rs_Priors, rs_Mu, rs_Sigma);

[rs_expData_gmr_norm, rs_beta] = GMR(rs_Priors, rs_Mu, rs_Sigma, rs_data_var_norm_test(1:nbVarInput,:),[1:nbVarInput],[nbVarAll]);
rs_expData_gmr = rs_expData_gmr_norm * std(y_train)+ mean(y_train);


%% RS Load prediction using GGMR
sum_beta_rs=sum(rs_beta,1);
[rs_Priors, rs_Mu, rs_Sigma, rs_expData_ggmr_norm] = Evolving_LW_2(rs_Priors, rs_Mu, rs_Sigma, rs_data_var_norm_test,sum_beta_rs);
rs_expData_ggmr_norm = rs_expData_ggmr_norm.';
rs_expData_ggmr= rs_expData_ggmr_norm*std(y_train)+mean(y_train); %Actual predicted flow after denormalization



%% Plot
% figure('position',[10,10,800,500],'name','GMR-GGMR-RS-Load Prediction');
% subplot(1,1,1); hold on;
% 
% xlabel('Time step, 5 min interval') 
% ylabel('Radiant Slab Loads (W)') 
% plot(rs_expData_gmr);
% plot(rs_expData_ggmr);
% plot(y_test);

rmse_gmr = (sum((rs_expData_gmr - y_test).^2) / length(y_test)).^ (0.5); 
mean_model_gmr = mean(abs(rs_expData_gmr));
std_model_gmr = (sum((rs_expData_gmr - mean_model_gmr).^2) / length(rs_expData_gmr)) .^ (0.5); 
nrmse_gmr = rmse_gmr *100 / std_model_gmr;
mean_measured_gmr = mean(abs(y_test));
cvrmse_gmr = rmse_gmr*100 / mean_measured_gmr;
mae_gmr = sum(abs(y_test - rs_expData_gmr)) / length(y_test);
mape_ratio_gmr = abs(y_test - rs_expData_gmr) ./ abs(y_test);
mape_ratio_gmr(isinf(mape_ratio_gmr)) = 0;
mape_gmr = sum(mape_ratio_gmr)*100 / length(y_test);

rmse_ggmr = (sum((rs_expData_ggmr - y_test).^2) / length(y_test)).^ (0.5); 
mean_model_ggmr = mean(abs(rs_expData_ggmr));
std_model_ggmr = (sum((rs_expData_ggmr - mean_model_ggmr).^2) / length(rs_expData_ggmr)) .^ (0.5); 
nrmse_ggmr = rmse_ggmr *100 / std_model_ggmr;
mean_measured_ggmr = mean(abs(y_test));
cvrmse_ggmr = rmse_ggmr*100 / mean_measured_ggmr;
mae_ggmr = sum(abs(y_test - rs_expData_ggmr)) / length(y_test);
mape_ratio_ggmr = abs(y_test - rs_expData_ggmr) ./ abs(y_test);
mape_ratio_ggmr(isinf(mape_ratio_ggmr)) = 0;
mape_ggmr = sum(mape_ratio_ggmr)*100 / length(y_test);
    
% title({"Left: GMR; Right: GGMR ",...
%     "NRMSE is " + nrmse_gmr + "%,  "+ nrmse_ggmr + "%",...
%     "CVRMSE is " + cvrmse_gmr + "%, "+ cvrmse_ggmr + "%",...
%     "MAE is " + mae_gmr + "W, "+ mae_ggmr + "W",...
%      "MAPE is " + mape_gmr + "%, "+ mape_ggmr + "%"})
%  
% legend({'GMR Predicted','GGMR Predicted','Actual'},'Location','southwest')