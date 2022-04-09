function [nrmse_gmr, cvrmse_gmr, mae_gmr, mape_gmr, ...
    nrmse_ggmr, cvrmse_ggmr, mae_ggmr, mape_ggmr] = ...
    gmr_ggmr_rs(nbStates, input_case, L_rate)
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
    
    q_solar = T{:,75}.*100;
    q_light = T{:,81}.*0.19;
    q_inte_heat = T{:,79}.*1.52;
    rc_y = T{:,82};
    valve_ht = T{:,83};
    valve_cl = T{:,84};
    
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
    valve_ht = valve_ht.';
    valve_cl = valve_cl.';
    y = y.';
    u_measure_table=readtable('data/u_arr_Tran.csv');
    u_measured = u_measure_table{:,:};

    fname = 'data/abcd.json'; 
    fid = fopen(fname); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    abcd = jsondecode(str);

    save('data/case_arr_sim.mat','t_slabs','t_cav','t_water_sup',...
    't_water_ret','vfr_water','q_solar','q_light','q_inte_heat',...
    'ahu_cfm1','ahu_t_sup1','ahu_cfm2','ahu_t_sup2','t_out',...
    'rc_y','valve_ht','valve_cl','y', "u_measured","abcd");

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load case data
load('data/case_arr_sim.mat'); %load 'Data'
% %%Correlation analysis
% to_do_corr = [t_out; t_slabs;t_cav;...
%     q_solar;q_light;q_inte_heat;ahu_cfm1;ahu_t_sup1;ahu_cfm2;...
%     ahu_t_sup2;valve_ht;valve_cl;rc_y;y];
% to_do_corr = to_do_corr.';
% corrcoefs = corrcoef(to_do_corr);
% 
% xvalues = {'t_out','t_slabs','t_cav','q_solar','q_light','q_inte_heat',...
%     'ahu_cfm1','ahu_t_sup1','ahu_cfm2','ahu_t_sup2',...
%     'valve_ht','valve_cl','rc_y','y'};
% yvalues = {'t_out','t_slabs','t_cav','q_solar','q_light','q_inte_heat',...
%     'ahu_cfm1','ahu_t_sup1','ahu_cfm2','ahu_t_sup2',...
%     'valve_ht','valve_cl','rc_y','y'};
% heat_coef = heatmap(xvalues, yvalues, corrcoefs);

total_length = size(y,2);
training_length = 4032;
test_initial_time = training_length -1;
rc_warming_step = 14;
testing_length = total_length - training_length;
% testing_length = 1000;

talk_to_rc  = 0;
with_predicted_flow = 0;

switch (input_case)
    case 1 %This is reserved case
        All_Variables = [t_out; t_slabs;t_cav;...
           valve_ht;valve_cl;q_solar;y];
    case 2 %This is reserved case with RC_y info
        talk_to_rc  = 1;
        All_Variables = [t_out; t_slabs;t_cav;...
           valve_ht;valve_cl;q_solar;rc_y;y];
    case 3
        with_predicted_flow = 1;
        All_Variables = [t_out; t_slabs;t_cav;...
           valve_ht;valve_cl;vfr_water; y];
    case 4
        talk_to_rc  = 1;
        with_predicted_flow = 1;
         All_Variables = [t_out; t_slabs;t_cav;...
           valve_ht;valve_cl;vfr_water;rc_y; y];
     case 5 
        All_Variables = [t_out; t_slabs;t_cav;y];
     case 6 
        All_Variables = [t_out; t_slabs;t_cav; ...
            valve_ht;valve_cl; y];
%         All_Variables = [t_out; t_slabs;t_cav;...
%             q_solar;q_light;q_inte_heat;ahu_cfm1;ahu_t_sup1;ahu_cfm2;...
%             ahu_t_sup2;valve_ht;valve_cl;rc_y;y];
end


nbVarAll = size(All_Variables,1);
nbVarInput = nbVarAll - 1;

for idx = 1:nbVarAll
    cur_var_train = All_Variables(idx,1:training_length);
    [cur_var_train_norm, cur_var_train_c, cur_var_train_s] = normalize(cur_var_train);
    cur_var_test = All_Variables(idx,training_length+1 :training_length+testing_length);
    cur_var_test_norm = (cur_var_test - cur_var_train_c) ./ cur_var_train_s;
    train(idx, :) =  cur_var_train;
    test(idx, :) =  cur_var_test;
    train_norm(idx, :) =  cur_var_train_norm;
    test_norm(idx, :) =  cur_var_test_norm;
end

%% Predict Flow Information
if with_predicted_flow == 1
    flow_talk_to_rc = 1;
    flow_All_Variables = [t_out; t_slabs;t_cav;...
               valve_ht;valve_cl;rc_y;vfr_water];
    flow_nbVarAll = size(flow_All_Variables,1);
    flow_nbVarInput = flow_nbVarAll - 1;
    
    for idx = 1:flow_nbVarAll
        flow_cur_var_train = flow_All_Variables(idx,1:training_length);
        [flow_cur_var_train_norm, flow_cur_var_train_c, flow_cur_var_train_s]...
            = normalize(flow_cur_var_train);
        flow_cur_var_test = flow_All_Variables(idx,training_length+1 :training_length+testing_length);
        flow_cur_var_test_norm = (flow_cur_var_test - flow_cur_var_train_c) ./ flow_cur_var_train_s;
        flow_train(idx, :) =  flow_cur_var_train;
        flow_test(idx, :) =  flow_cur_var_test;
        flow_train_norm(idx, :) =  flow_cur_var_train_norm;
        flow_test_norm(idx, :) =  flow_cur_var_test_norm;
    end
    
    [flow_Priors, flow_Mu, flow_Sigma] = EM_init_kmeans(flow_train_norm, nbStates);
    [flow_Priors, flow_Mu, flow_Sigma]  = EM(flow_train_norm, flow_Priors, flow_Mu, flow_Sigma);
    [flow_expData_gmr_norm, flow_beta] = GMR(flow_Priors, flow_Mu, flow_Sigma, ...
        flow_test_norm(1:flow_nbVarInput,:),[1:flow_nbVarInput],[flow_nbVarAll]);

    sum_beta_flow=sum(flow_beta,1);
    center_flow_y = mean(flow_train(flow_nbVarAll - 1,:));
    scale_flow_y = std(flow_train(flow_nbVarAll - 1,:));
    [flow_Priors, flow_Mu, flow_Sigma, flow_expData_ggmr_norm] = ...
    Evolving_LW_2(flow_Priors, flow_Mu, flow_Sigma, flow_test_norm,...
    sum_beta_flow,flow_talk_to_rc, test_initial_time, center_flow_y, scale_flow_y,...
    u_measured, rc_warming_step,abcd,L_rate);

    disp("updating flow")
    test_norm(nbVarInput - talk_to_rc,:) = flow_expData_ggmr_norm;

end

%% RS Load prediction using GMR
center_rc_y = mean(train(nbVarAll - 1,:));
scale_rc_y = std(train(nbVarAll - 1,:));

[rs_Priors, rs_Mu, rs_Sigma] = EM_init_kmeans(train_norm, nbStates);
[rs_Priors, rs_Mu, rs_Sigma]  = EM(train_norm, rs_Priors, rs_Mu, rs_Sigma);

if talk_to_rc == 1
    for t = 1:size(test_norm,2)
        target_time = t + test_initial_time;
        u_arr = u_measured(:,target_time + 1- rc_warming_step:target_time+1);
        result = RC_PredictedRealTime(u_arr,abcd);
%         result = communication(target_time);
%         real_y(1,t) = result;
%         real_new_y(1,t) = result_new;
%         real_err = result - test(nbVarAll,t)
%         one_err = test(nbVarInput,t) - test(nbVarAll,t)
        result_norm = (result - center_rc_y) /  scale_rc_y;
        test_norm(nbVarAll-1,t) = result_norm;
    end
end

[rs_expData_gmr_norm, rs_beta] = GMR(rs_Priors, rs_Mu, rs_Sigma, ...
    test_norm(1:nbVarInput,:),[1:nbVarInput],[nbVarAll]);

rs_expData_gmr = rs_expData_gmr_norm * std(train(nbVarAll,:))+ mean(train(nbVarAll,:));


%% RS Load prediction using GGMR
sum_beta_rs=sum(rs_beta,1);


[rs_Priors, rs_Mu, rs_Sigma, rs_expData_ggmr_norm] = ...
    Evolving_LW_2(rs_Priors, rs_Mu, rs_Sigma, test_norm,...
    sum_beta_rs,talk_to_rc, test_initial_time, center_rc_y, scale_rc_y,...
    u_measured, rc_warming_step,abcd,L_rate);
rs_expData_ggmr_norm = rs_expData_ggmr_norm.';
rs_expData_ggmr= rs_expData_ggmr_norm*std(train(nbVarAll,:))+mean(train(nbVarAll,:)); %Actual predicted flow after denormalization



%% Plot
y_test = test(nbVarAll,:);

%%⬇️To hourly
test_hours = fix(size(y_test,2) / 12);
y_test = reshape(y_test, 12, test_hours );
y_test = sum(y_test);

rs_expData_gmr = reshape(rs_expData_gmr, 12, test_hours );
rs_expData_gmr = sum(rs_expData_gmr);

rs_expData_ggmr = reshape(rs_expData_ggmr, 12, test_hours );
rs_expData_ggmr = sum(rs_expData_ggmr);
%%⬆️To hourly

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