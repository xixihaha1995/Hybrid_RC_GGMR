import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import _2_all_models as all_predict
import numpy as np
nbStates = 15
L_rate = 5e-3

All_Variables_obj, u_measured_obj, abcd_obj = general_tools.switch_case(0)
All_Variables = All_Variables_obj.astype('float64')
u_measured = u_measured_obj.astype('float64')
abcd = abcd_obj

total_length = All_Variables.shape[1]
training_length = 4032
test_initial_time = training_length
rc_warming_step = 14
testing_length = total_length - training_length
# testing_length = 1000


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
'''GGMR'''
ggmr_norm = all_predict.GGMR_prediction(train_norm, test_norm, nbStates)
ggmr_predict = general_tools.de_norm(ggmr_norm, scale_y, center_y)
cvrmse_ggmr = general_tools.cvrmse_cal(y_test,ggmr_predict,mean_measured)
'''Hybrid No Flow Information'''
hybrid_norm = all_predict.Hybrid_prediction(train_norm, test_norm, nbStates, nbVarInput,test_initial_time,
                      center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd,L_rate)
hybrid_predict = general_tools.de_norm(hybrid_norm, scale_y, center_y)
cvrmse_hybrid = general_tools.cvrmse_cal(y_test,hybrid_predict,mean_measured)
'''Save results'''
results = {}
results['y_test'] = y_test.tolist()
results['rc_y'] = rc_y.tolist()
results['cvrmse_rc'] = cvrmse_rc
results['ggmr_predict'] = ggmr_predict.tolist()
results['cvrmse_ggmr'] = cvrmse_ggmr
results['hybrid_predict'] = hybrid_predict.tolist()
results['cvrmse_hybrid'] = cvrmse_hybrid
general_tools.saveJSON(results, "results")


