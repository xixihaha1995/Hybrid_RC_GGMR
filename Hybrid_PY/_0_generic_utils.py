import os, json, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

def loadJSON(name):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'inputs',name + '.json'), 'r') as f:
        testDict = json.loads(f.read())
    return testDict

def loadJSONFromOutputs(name):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'outputs',name + '.json'), 'r') as f:
        testDict = json.loads(f.read())
    return testDict

def saveJSON(data, name):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'outputs',name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
    return case_arr, measured_modeled_arr, u_arr_Tran_arr, abcd

def switch_case(case_nb):
    pass
    case_arr, measured_modeled_arr, u_measured, abcd = load_csv()
    t_out = (case_arr[:, 0] - 32) * 5 / 9
    t_slab = ((case_arr[:, 5] + case_arr[:, 6] + case_arr[:, 7] + case_arr[:, 8] + case_arr[:, 9]+ case_arr[:, 10]
               + case_arr[:, 11] + case_arr[:, 12] + case_arr[:, 13] + case_arr[:,14]) / 10- 32) * 5 / 9
    vfr_water = case_arr[:, 27]
    t_cav = (case_arr[:, 49] - 32) * 5 / 9
    valve_ht = case_arr[:,82]
    valve_cl = case_arr[:,83]

    rc_y = measured_modeled_arr[:, 1]
    y = measured_modeled_arr[:, 0]



    if case_nb == 0:
        All_Variables = np.array([t_out, t_slab, t_cav,valve_ht,valve_cl, rc_y, y])
    if case_nb == 1:
        All_Variables = np.array([t_out,t_slab, t_cav,valve_ht,valve_cl, rc_y, vfr_water])
    if case_nb == 2:
            noise = np.random.normal(0, 100, y.shape[0])
            fake_rc_y = y + noise
            All_Variables = np.array([t_out, t_slab, t_cav, valve_ht, valve_cl,vfr_water, rc_y, y])
    return All_Variables, u_measured, abcd

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

def cvrmse_cal(measure, predict, mean_measured):
    rmse = (sum((measure - predict) ** 2) / len(measure)) ** (1 / 2)
    cvrmse = rmse * 100 / mean_measured
    return cvrmse

def de_norm(norm, scale_y, center_y):
    norm_tmp = np.array(norm).reshape(-1)
    predict_temp = norm_tmp * scale_y + center_y
    predict = predict_temp.reshape(-1)
    return predict

def ggmr_load_all_var(training_length,testing_length):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    case_arr_abs = os.path.join(script_dir, 'inputs', 'ggmr_all_in_one.csv')
    df  =  pd.read_csv(case_arr_abs)
    df.Timestamp = pd.to_datetime(df.Timestamp)
    df['minute'] = df.apply(lambda x: x['Timestamp'].minute, axis=1)
    df['hour'] = df.apply(lambda x: x['Timestamp'].hour, axis=1)
    df['dayofweek'] = df.apply(lambda x: x['Timestamp'].dayofweek, axis=1)
    new_cols = ['dayofweek','hour','minute','t_out','t_slab1',
                't_cav','valve_ht','valve_cl','vfr_water','rc_y','y']

    df = df.sort_values("Timestamp").drop("Timestamp", axis=1)
    # new_cols = ['t_out','t_slab1','t_cav','valve_ht',
    #             'valve_cl','vfr_water','rc_y','y']
    df = df[new_cols]
    train_df = df.iloc[:training_length,:]
    test_df = df.iloc[training_length:training_length +testing_length, :]

    train_ori = train_df.values
    test_ori = test_df.values

    feature_sc = MinMaxScaler()
    label_sc = MinMaxScaler()

    scale_both = False
    if not scale_both:
        feature_sc.fit(train_ori)
        train_scaled = feature_sc.transform(train_ori)
        test_scaled = feature_sc.transform(test_ori)
    else:
        feature_sc.fit(df.values)
        train_scaled = feature_sc.transform(train_ori)
        test_scaled = feature_sc.transform(test_ori)

    label_sc.fit(train_df.iloc[:, -1].values.reshape(-1, 1))
    return label_sc, train_scaled, test_scaled, train_ori, test_ori