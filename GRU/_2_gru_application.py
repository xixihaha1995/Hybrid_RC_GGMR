import numpy as np, torch
import _1_gru_pack as gru_tools, _0_tools as general_tool
'''Configurations'''
data_dir = "data/"
test_length = 10860
drop_rc_y = False
bidirectional = False

lookback = 5
_hidden_dim=256 #200
_epoch =500 # 50
lr =  1e-3 #1e-3
drop_out_prob = 0.2
batch_size = 1024
device = torch.device("cpu")
'''Preprocessing'''
label_scalers, train_data, train_loader, test_x, test_y = gru_tools.preprocess(data_dir, lookback,
                                                                               test_length, batch_size, drop_rc_y)
'''Train and evaluate'''
gru_model = gru_tools.train(train_loader, lr,batch_size, _hidden_dim,_epoch,device, model_type="GRU")
gru_outputs, targets, gru_sMAPE = gru_tools.evaluate(gru_model, test_x, test_y, label_scalers, device)
gru_outputs, targets = np.array(gru_outputs), np.array(targets)
'''Save results'''
rc_y_all, y_measured_all = general_tool.load_rc_y_y()
rc_y_test, y_measured_test = rc_y_all[-gru_outputs.shape[1]:],  y_measured_all[-gru_outputs.shape[1]:]
mean_measured = abs(y_measured_test).mean()
cvrmse_rc = general_tool.cvrmse_cal(y_measured_test,rc_y_test,mean_measured)
cvrmse_gru = general_tool.cvrmse_cal(targets.reshape(-1),gru_outputs.reshape(-1),mean_measured)

results = {}
results['y_test'] = targets.reshape(-1).tolist()
results['rc_y'] = rc_y_test.tolist()
results['cvrmse_rc'] = cvrmse_rc
results['gru_outputs'] = gru_outputs.reshape(-1).tolist()
results['cvrmse_gru'] = cvrmse_gru

general_tool.saveJSON(results, "_results_gru")
