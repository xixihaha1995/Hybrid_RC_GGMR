import numpy as np
import pandas as pd
import _1_utils, csv
from scipy import signal

u_train = None
y_train = None
u_test = None
y_test = None

load_u_y = False

def call_load_u_y(constants):
    global u_train, y_train, u_test, y_test, load_u_y
    (u_train, y_train) = _1_utils.load_u_y(constants)
    (u_test, y_test) = _1_utils.load_u_y(constants, train=False)
    load_u_y = True


def init_pos(case_nbr, n_particles):
    with open('./init_rscs.txt', 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        for row in reader:
            row = [float(s_num) for s_num in row]
            rscs_str.append(row)
    rscs_lst = rscs_str[case_nbr]
    rscs_init = np.array([rscs_lst for _ in range(n_particles)])
    return rscs_init


def paras_to_ABCD_swarm(params, constants):

    A_init = np.zeros((constants['state_num'], constants['state_num']))
    B_init = np.zeros((constants['state_num'], constants['input_num']))
    C_init = np.zeros((1, constants['state_num']))
    D_init = np.zeros((1, constants['input_num']))

    A, B, C, D = _1_utils.assgin_ABCD(A_init, B_init, C_init, D_init, params, case_nbr = constants['case_nbr'])

    sys = signal.StateSpace(A, B, C, D)
    sys_d = sys.to_discrete(constants['ts_sampling'])
    a = sys_d.A
    b = sys_d.B
    c = sys_d.C
    d = sys_d.D

    return a, b, c, d


def obj_func(params, constants, train=True):
    if not load_u_y:
        call_load_u_y(constants)
    if train:
        u_arr, y_arr = u_train, y_train
    else:
        u_arr, y_arr = u_test, y_test
    a, b, c, d = paras_to_ABCD_swarm(params, constants)
    y_model = np.zeros_like(y_arr)
    if constants['case_nbr'] == 3:
        x_discrete = np.array([[11], [22], [22], [25], [25]])
    elif constants['case_nbr'] == 0:
        x_discrete = np.array([[22]])
    elif constants['case_nbr'] == -1:
        x_discrete = 25 * np.ones((constants['state_num'], 1))
    state_num = constants['state_num']
    for i in range(u_arr.shape[1]):
        y_model[i,] = c @ x_discrete + d @ u_arr[:, i]
        x_discrete = a @ x_discrete + (b @ u_arr[:, i]).reshape((state_num, 1))

    return y_arr, y_model


def particle_loss(params, constants):
    y_measure, y_model = obj_func(params, constants)
    return sum((abs(y_model - y_measure))**2)


def whole_swarm_loss(x, constants):
    n_particles = x.shape[0]
    j = [particle_loss(x[i], constants) for i in range(n_particles)]
    return np.array(j)


def predict(pos, constants):
    y_train, y_train_pred = obj_func(pos, constants, train=True)
    ytest, y_test_pred = obj_func(pos, constants, train=False)
    return y_train, y_train_pred, ytest, y_test_pred


def load_pos():
    with open('./pos_rscs.txt', 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        all_pos = []
        for row in reader:
            cur_pos_dict = {}
            cur_pos_dict['title'] = row[0]
            cur_pos = []
            for j in range(1, len(row)):
                cur_pos.append(float(row[j]))
            cur_pos_dict['pos'] = cur_pos
            all_pos.append(cur_pos_dict)
    return all_pos

