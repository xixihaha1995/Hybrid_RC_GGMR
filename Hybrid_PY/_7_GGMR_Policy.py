import _0_generic_utils as general_tools, _1_1_ggmr_utils
import _2_all_models as all_predict
import numpy as np
'''Configuration/Hyper-parameters'''
training_length = 4032
testing_length = 200
nbStates = 8
L_rate = 1e-1 #policy one
T_merge = 1e4 #policy two

look_back_batch_size, predict_size = 2, 1
_hybrid = True
plonum = 2
'''Preprocessing (fit_transform based on training data, saved scaler)'''
label_sc, train_scaled, test_scaled, train_ori, test_ori = \
    general_tools.ggmr_load_all_var(training_length,testing_length)
'''Hyper-parameters tunning'''
results = {}
train_scaled_trans, test_scaled_trans = train_scaled.T, test_scaled.T
train_ori_trans, test_ori_trans = train_ori.T, test_ori.T
ggmr_norm = _1_1_ggmr_utils.online_ggmr_new_dominate(train_scaled_trans, test_scaled_trans,
                                                       train_ori_trans, test_ori_trans ,
                                                       nbStates, L_rate,
                look_back_batch_size, predict_size,_hybrid, T_merge, plonum)

# ggmr_norm = all_predict.GGMR_prediction(train_scaled_trans, test_scaled_trans, nbStates, T_Sigma, L_rate)
ggmr_predict = label_sc.inverse_transform(np.array(ggmr_norm).reshape(-1, 1))
ggmr_predict = ggmr_predict.reshape(-1)
pass
'''Results'''
y_test = test_ori.T[-1,:]
rc_y = test_ori.T[-2,:]
mean_measured = abs(y_test).mean()
cvrmse_rc = general_tools.cvrmse_cal(y_test,rc_y, mean_measured)
cvrmse_ggmr = general_tools.cvrmse_cal(y_test, ggmr_predict, mean_measured)

results['y_test'] = y_test.tolist()
results['rc_y'] = rc_y.tolist()
results['ggmr_predict'] = ggmr_predict.tolist()
results['cvrmse_rc'] = cvrmse_rc
results['cvrmse_ggmr'] = cvrmse_ggmr
general_tools.saveJSON(results, "_results_scale_policy")




