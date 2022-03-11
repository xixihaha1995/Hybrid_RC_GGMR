import numpy as np
import pandas as pd
import _1_utils, csv
from scipy import signal


def init_pos(n_particles):
    with open('./init_rscs.csv', 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        for row in reader:
            rscs_str.append(row)
    rscs_str = rscs_str[0]
    rscs_lst = [float(s_num) for s_num in rscs_str]
    rscs_lst.insert(0, 0)
    rscs_init = np.array([rscs_lst for _ in range(n_particles)])
    return rscs_init


def paras_to_ABCD_swarm(params, constants):

    A_init = np.zeros((constants['state_num'], constants['state_num']))
    B_init = np.zeros((constants['state_num'], constants['input_num']))
    C_init = np.zeros((1, constants['state_num']))
    D_init = np.zeros((1, constants['input_num']))

    A, B, C, D = _1_utils.assgin_ABCD(A_init, B_init, C_init, D_init, params)

    sys = signal.StateSpace(A, B, C, D)
    sys_d = sys.to_discrete(constants['ts_sampling'])
    a = sys_d.A
    b = sys_d.B
    c = sys_d.C
    d = sys_d.D

    return a, b, c, d


def obj_func(params, constants, train=True):
    if train:
        (u_arr, y_arr) = _1_utils.load_u_y(constants)
    else:
        (u_arr, y_arr) = _1_utils.load_u_y(constants, train=False)
    a, b, c, d = paras_to_ABCD_swarm(params, constants)
    y_model = np.zeros_like(y_arr)
    x_discrete = 25 * np.ones((7, 1))
    state_num = constants['state_num']
    for i in range(u_arr.shape[1]):
        y_model[i,] = c @ x_discrete + d @ u_arr[:, i]
        x_discrete = a @ x_discrete + (b @ u_arr[:, i]).reshape((state_num, 1))

    return y_arr, y_model


def particle_loss(params, constants):
    y_measure, y_model = obj_func(params, constants)
    return sum((abs(y_model - y_measure))**2) /(constants['end'] - constants['start'])


def whole_swarm_loss(x, constants):
    n_particles = x.shape[0]
    j = [particle_loss(x[i], constants) for i in range(n_particles)]
    return np.array(j)


def predict(pos, constants):
    y_train, y_train_pred = obj_func(pos, constants, train=True)
    ytest, y_test_pred = obj_func(pos, constants, train=False)
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
