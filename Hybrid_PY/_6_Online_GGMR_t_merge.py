import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import _2_all_models as all_predict
import numpy as np

max_nbStates = 15
_look_back_batch_size = 5
_predict_size = 5
_hybrid = True
t_merge_candidates = [1e5, 1e6, 1e7]
L_rate = 1e-1

training_length = 4032
rc_warming_step = 15

All_Variables_obj, u_measured_obj, abcd = general_tools.switch_case(2)
All_Variables = All_Variables_obj.astype('float64')
u_measured = u_measured_obj.astype('float64')
total_length = All_Variables.shape[1]
test_initial_time = training_length
testing_length = total_length - training_length
testing_length = 500
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
updated_flow_res = general_tools.loadJSONFromOutputs("_flow_norm_results")
hybrid_flow_norm = updated_flow_res['hybrid_flow_norm']
test_norm[-3,:] = np.array(hybrid_flow_norm)[:testing_length]

'''Hyper-parameters tunning'''
results = {}
for t_merge in t_merge_candidates:
    '''GGMR'''
    ggmr_norm = gaussian_tools.online_ggmr(train_norm,test_norm,max_nbStates,L_rate,t_merge,
                                           _look_back_batch_size, _predict_size, _hybrid)
    ggmr_predict = general_tools.de_norm(ggmr_norm, scale_y, center_y)
    cvrmse_ggmr = general_tools.cvrmse_cal(y_test, ggmr_predict, mean_measured)
    results[f'ggmr_predict_{t_merge:.6f}'] = ggmr_predict.tolist()
    results[f'cvrmse_ggmr_{t_merge:.6f}'] = cvrmse_ggmr
'''Save results'''
results['y_test'] = y_test.tolist()
results['rc_y'] = rc_y.tolist()
results['cvrmse_rc'] = cvrmse_rc
general_tools.saveJSON(results, "_results_online_t_merge")




