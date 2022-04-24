import os, json, pandas as pd, numpy as np

def loadJSON(name):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'inputs',name + '.json'), 'r') as f:
        testDict = json.loads(f.read())
    return testDict

def load_csv():
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    case_arr_abs = os.path.join(script_dir, 'inputs', 'case_arr.csv')
    _measured_modeled_abs = os.path.join(script_dir, 'inputs', '6_measured_modeled.csv')
    u_arr_Tran_abs = os.path.join(script_dir, 'inputs', 'u_arr_Tran.csv')
    abcd = loadJSON("abcd")

    case_csv = pd.read_csv(case_arr_abs, header= None)
    _measured_modeled = pd.read_csv(_measured_modeled_abs)
    u_arr_Tran = pd.read_csv(u_arr_Tran_abs, header=None)

    case_arr = case_csv.to_numpy()
    measured_modeled_arr = _measured_modeled.to_numpy()
    u_arr_Tran_arr = u_arr_Tran.to_numpy()
    return case_arr, measured_modeled_arr, u_arr_Tran_arr

def switch_case(case_nb):
    pass
    case_arr, measured_modeled_arr, u_arr_Tran_arr = load_csv()
    t_out = (case_arr[:, 0] - 32) * 5 / 9
    t_slab = ((case_arr[:, 5] + case_arr[:, 6] + case_arr[:, 7] + case_arr[:, 8] + case_arr[:, 9]+ case_arr[:, 10]
               + case_arr[:, 11] + case_arr[:, 12] + case_arr[:, 13] + case_arr[:,14]) / 10- 32) * 5 / 9
    t_cav = (case_arr[:, 49] - 32) * 5 / 9
    valve_ht = case_arr[:,82]
    valve_cl = case_arr[:,83]
    rc_y = measured_modeled_arr[:, 1]
    y = measured_modeled_arr[:, 0]
    if case_nb == 0:
        All_Variables = np.array([t_out, t_slab, t_cav,valve_ht,valve_cl, rc_y, y])
    return All_Variables

def split_train_test_norm(nbVarAll, All_Variables,training_length,testing_length):
    train_lst, test_lst = [], []
    train_norm_lst, test_norm_lst = [], []
    for idx in range(nbVarAll):
        cur_var_train = All_Variables[idx, :training_length]
        cur_var_train_c, cur_var_train_s = cur_var_train.mean(), cur_var_train.std()
        cur_var_train_norm = (cur_var_train - cur_var_train_c) / cur_var_train_s
        cur_var_test = All_Variables[idx, training_length:training_length + testing_length]
        cur_var_test_c, cur_var_test_s = cur_var_test.mean(), cur_var_test.std()
        cur_var_test_norm = (cur_var_test - cur_var_test_c) / cur_var_test_s
        train_lst.append(cur_var_train)
        test_lst.append(cur_var_test)
        train_norm_lst.append(cur_var_train_norm)
        test_norm_lst.append(cur_var_test_norm)
    train = np.array(train_lst)
    test = np.array(test_lst)
    train_norm = np.array(train_norm_lst)
    test_norm = np.array(test_norm_lst)
    return train, test, train_norm, test_norm