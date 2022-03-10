import numpy as np
import pandas as pd
import _1_config, _2_utils, csv
from scipy import signal


def init_pos():
    with open('./init_rscs.csv', 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        for row in reader:
            rscs_str.append(row)
    rscs_str = rscs_str[0]
    rscs_lst = [float(s_num) for s_num in rscs_str]
    # rscs_lst.insert(0, 0)
    rscs_init = np.array([rscs_lst for _ in range(_1_config.n_particles)])
    return rscs_init


def paras_to_ABCD_swarm(params):
    A_init = np.zeros((_1_config.state_num, _1_config.state_num))
    B_init = np.zeros((_1_config.state_num, _1_config.input_num))
    C_init = np.zeros((1, _1_config.state_num))
    D_init = np.zeros((1, _1_config.input_num))

    A, B, C, D = _2_utils.assgin_ABCD(A_init, B_init, C_init, D_init, params)

    sys = signal.StateSpace(A, B, C, D)
    sys_d = sys.to_discrete(_1_config.ts_sampling)
    a = sys_d.A
    b = sys_d.B
    c = sys_d.C
    d = sys_d.D

    return a, b, c, d


def obj_func(params, train=True):
    if train:
        (u_arr, y_arr) = _2_utils.load_u_y()
        _1_config.u_arr = u_arr
        _1_config.y_arr = y_arr
    else:
        (u_arr, y_arr) = _2_utils.load_u_y(train=False)
        _1_config.u_arr_test = u_arr
        _1_config.y_arr_test = y_arr

    a, b, c, d = paras_to_ABCD_swarm(params)
    y_model = np.zeros_like(y_arr)
    # x_discrete = 25 * np.ones((7, u_arr.shape[1]))
    x_discrete = 25 * np.ones((_1_config.state_num, 1))
    state_num = _1_config.state_num
    for i in range(u_arr.shape[1]):
        y_model[i,] = c @ x_discrete + d @ u_arr[:, i]
        x_discrete = a @ x_discrete + (b @ u_arr[:, i]).reshape((state_num, 1))

    return y_arr, y_model


def particle_loss(params):
    y_measure, y_model = obj_func(params)
    return sum(abs(y_model - _1_config.y_arr))


def whole_swarm_loss(x):
    n_particles = x.shape[0]
    j = [particle_loss(x[i]) for i in range(n_particles)]
    return np.array(j) / _1_config.u_arr.shape[1]


def predict(pos):
    y_train, y_train_pred = obj_func(pos, train=True)
    ytest, y_test_pred = obj_func(pos, train=False)
    return y_train, y_train_pred, ytest, y_test_pred


def load_pos():
    with open('./pos_rscs.csv', 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        for row in reader:
            rscs_str.append(row[0].split('\t'))
    all_pos = []
    for i in range(len(rscs_str)):
        cur_pos_dict = {}
        cur_pos_dict['title'] = rscs_str[i][0]
        cur_pos = []
        for j in range(1, len(rscs_str[i])):
            cur_pos.append(float(rscs_str[i][j]))
        cur_pos.insert(0, 0)
        cur_pos_dict['pos'] = cur_pos
        all_pos.append(cur_pos_dict)
    return all_pos
