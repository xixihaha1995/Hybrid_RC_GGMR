import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import _2_all_models as all_predict
import numpy as np

nbStates = 5
L_rate = 5e-3
T_Sigma = 5
nbStates_candidates = [5,10,15] #5
T_Sigma_candidates = [2e-1, 2, 5] #2
L_rate_candidates = [1e-3, 3e-2, 6e-2]
training_length = 4032
rc_warming_step = 15

All_Variables_obj, u_measured_obj, abcd = general_tools.switch_case(2)
All_Variables = All_Variables_obj.astype('float64')
u_measured = u_measured_obj.astype('float64')
total_length = All_Variables.shape[1]
test_initial_time = training_length
# testing_length = total_length - training_length
testing_length = 200
nbVarAll = All_Variables.shape[0]
nbVarInput = nbVarAll - 1

train, test, train_norm, test_norm = general_tools.split_train_test_norm(
    nbVarAll, All_Variables,training_length, testing_length)
center_rc_y, scale_rc_y = train[-2,:].mean(), train[-2,:].std()
'''De-normalization'''
y_test = test[-1,:]
rc_y = test[-2,:]
mean_measured = abs(y_test).mean()
center_y, scale_y = train[-1,:].mean(), train[-1,:].std()
cvrmse_rc = general_tools.cvrmse_cal(y_test,rc_y,mean_measured)
'''Update flow info in test_norm with minor Hybrid approach'''
# updated_flow_res = general_tools.loadJSONFromOutputs("_flow_norm_results")
# hybrid_flow_norm = updated_flow_res['hybrid_flow_norm']
# test_norm[-3,:] = np.array(hybrid_flow_norm)
'''GGMR'''
ggmr_norm = all_predict.GGMR_prediction(train_norm, test_norm, nbStates, T_Sigma)
ggmr_predict = general_tools.de_norm(ggmr_norm, scale_y, center_y)
cvrmse_ggmr = general_tools.cvrmse_cal(y_test, ggmr_predict, mean_measured)
'''Hyper-parameters tunning'''
results = {}
for L_rate in L_rate_candidates:
    '''Hybrid'''
    hybrid_norm = all_predict.Hybrid_prediction(train_norm, test_norm, nbStates, nbVarInput,test_initial_time,
                          center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd,L_rate,T_Sigma)
    hybrid_predict = general_tools.de_norm(hybrid_norm, scale_y, center_y)
    cvrmse_hybrid = general_tools.cvrmse_cal(y_test,hybrid_predict,mean_measured)
    results[f'hybrid_predict_{L_rate:.3f}'] = hybrid_predict.tolist()
    results[f'cvrmse_hybrid_{L_rate:.3f}'] = cvrmse_hybrid
'''Save results'''
results['y_test'] = y_test.tolist()
results['rc_y'] = rc_y.tolist()
results['cvrmse_rc'] = cvrmse_rc
results[f'ggmr_predict'] = ggmr_predict.tolist()
results[f'cvrmse_ggmr'] = cvrmse_ggmr
general_tools.saveJSON(results, "_results_l_rate")
# general_tools.saveJSON(results, "_results_t_sigma")
# general_tools.saveJSON(results, "_nbStates_results")



