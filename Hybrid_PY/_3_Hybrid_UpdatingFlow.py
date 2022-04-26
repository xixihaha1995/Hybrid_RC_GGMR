import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import _2_all_models as all_predict
import numpy as np

nbStates = 15
L_rate = 5e-3
training_length = 4032
rc_warming_step = 15

All_Variables_obj, u_measured_obj, abcd = general_tools.switch_case(1)
All_Variables = All_Variables_obj.astype('float64')
u_measured = u_measured_obj.astype('float64')
total_length = All_Variables.shape[1]
test_initial_time = training_length
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
'''Hybrid No Flow Information'''
hybrid_flow_norm = all_predict.Hybrid_prediction(train_norm, test_norm, nbStates, nbVarInput,test_initial_time,
                      center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd,L_rate)
'''Save results'''
results = {}
results['hybrid_flow_norm'] = np.array(hybrid_flow_norm).reshape(-1).tolist()
general_tools.saveJSON(results, "_flow_norm_results")


