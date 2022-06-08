import _0_utils as general_tool, numpy as np

measure, hybrid, ggmr, rcm3, rcm2,rcm1 = general_tool.load_predicted_measure("all_models_save_5min.csv")
measure_hourly_all = general_tool.to_hourly(measure)
pass
# rc_model1_predict_hourly_all = general_tool.to_hourly(rc_model1_predict)
#
# rc_model2_predict_hourly_test, rc_model1_predict_hourly_test = \
#     rc_model2_predict_hourly_all[-905:],rc_model1_predict_hourly_all[-905:]
#
# np.savetxt('rc_model2_save.csv',rc_model2_predict_hourly_test)
# np.savetxt('rc_model1_save.csv',rc_model1_predict_hourly_test)