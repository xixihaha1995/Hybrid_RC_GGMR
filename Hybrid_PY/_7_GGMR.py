import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import _2_all_models as all_predict
import numpy as np
'''Configuration/Hyper-parameters'''
training_length = 4032
testing_length = 500
nbStates = 5
L_rate = 5e-3
T_Sigma = 5
'''Preprocessing (fit_transform based on training data, saved scaler)'''
label_sc, train_scaled, test_scaled, train_ori, test_ori = \
    general_tools.ggmr_load_all_var(training_length,testing_length)
'''Hyper-parameters tunning'''
results = {}
train_scaled_trans, test_scaled_trans = train_scaled.T, test_scaled.T
ggmr_norm = all_predict.GGMR_prediction(train_scaled_trans, test_scaled_trans, nbStates, T_Sigma, L_rate)
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
general_tools.saveJSON(results, "_results_initial_ggmr")




