import _0_utils as general_tool, numpy as np

rc_model2_predict, measure = general_tool.load_predicted_measure("7_measured_modeled.csv")
rc_model1_predict, measure = general_tool.load_predicted_measure("5_measured_modeled.csv")

rc_model2_predict_hourly_all = general_tool.to_hourly(rc_model2_predict)
rc_model1_predict_hourly_all = general_tool.to_hourly(rc_model1_predict)

rc_model2_predict_hourly_test, rc_model1_predict_hourly_test = \
    rc_model2_predict_hourly_all[-905:],rc_model1_predict_hourly_all[-905:]

np.savetxt('rc_model2_save.csv',rc_model2_predict_hourly_test)
np.savetxt('rc_model1_save.csv',rc_model1_predict_hourly_test)