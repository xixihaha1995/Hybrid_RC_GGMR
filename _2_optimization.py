import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit, csv
from scipy import signal

import _0_config


def assgin_ABCD(A, B, C, D, p):
    A[1 - 1, 1 - 1] = -(1 / (p[1] * p[12])) - (1 / (p[2] * p[12]))
    A[1 - 1, 2 - 1] = (1 / (p[2] * p[12]))
    A[2 - 1, 1 - 1] = 1 / (p[2] * p[13])
    A[2 - 1, 2 - 1] = -(1 / (p[2] * p[13]) + 1 / (p[3] * p[13]))
    A[3 - 1, 3 - 1] = -(1 / (p[4] * p[14]) + 1 / (p[5] * p[14]))
    A[3 - 1, 4 - 1] = 1 / (p[5] * p[14])
    A[4 - 1, 3 - 1] = 1 / (p[5] * p[15])
    A[4 - 1, 4 - 1] = -(1 / (p[5] * p[15]) + 1 / (p[6] * p[15]))
    A[5 - 1, 5 - 1] = -(1 / (p[7] * p[16]) + 1 / (p[8] * p[16]))
    A[6 - 1, 6 - 1] = -1 / (p[10] * p[18])
    A[6 - 1, 7 - 1] = 1 / (p[10] * p[18])
    A[7 - 1, 6 - 1] = 1 / (p[10] * p[19])
    A[7 - 1, 7 - 1] = -1 / (p[10] * p[19]) - 1 / (p[11] * p[19])

    B[1 - 1, 1 - 1] = (1 / (p[1] * p[12]))
    B[1 - 1, 3 - 1] = 1 / p[12]
    B[2 - 1, 4 - 1] = 1 / p[13]
    B[2 - 1, 11 - 1] = 1 / (p[3] * p[13])
    B[3 - 1, 1 - 1] = 1 / (p[4] * p[14])
    B[3 - 1, 5 - 1] = 1 / p[14]
    B[4 - 1, 6 - 1] = 1 / p[15]
    B[4 - 1, 11 - 1] = 1 / (p[6] * p[15])
    B[5 - 1, 2 - 1] = 1 / (p[7] * p[16])
    B[5 - 1, 7 - 1] = 1 / (p[16])
    B[5 - 1, 11 - 1] = 1 / (p[8] * p[16])
    B[6 - 1, 9 - 1] = p[21] / (2 * p[18])
    B[7 - 1, 9 - 1] = p[21] / (2 * p[19])
    B[7 - 1, 11 - 1] = 1 / (p[11] * p[19])

    C[1 - 1, 2 - 1] = -1 / p[3]
    C[1 - 1, 4 - 1] = -1 / p[6]
    C[1 - 1, 5 - 1] = -1 / p[8]
    C[1 - 1, 7 - 1] = -1 / p[11]

    D[1 - 1, 1 - 1] = -1 / p[9]
    D[1 - 1, 8 - 1] = -p[20]
    D[1 - 1, 10 - 1] = -p[22]
    D[1 - 1, 11 - 1] = 1 / p[3] + 1 / p[6] + 1 / p[8] + 1 / p[11] + 1 / p[9]
    D[1 - 1, 12 - 1] = p[17]

    return A, B, C, D


def paras_to_ABCD(params):
    p = [0] * 23
    p[1] = params['reout'].value
    p[2] = params['re'].value
    p[3] = params['rein'].value
    p[4] = params['rcout'].value
    p[5] = params['rc'].value
    p[6] = params['rcin'].value
    p[7] = params['rf'].value
    p[8] = params['rfin'].value
    p[9] = params['rw'].value
    p[10] = params['ri1'].value
    p[11] = params['ri2'].value
    p[12] = params['ce1'].value
    p[13] = params['ce2'].value
    p[14] = params['cc1'].value
    p[15] = params['cc2'].value
    p[16] = params['cf'].value
    p[17] = params['cair'].value
    p[18] = params['ci1'].value
    p[19] = params['ci2'].value
    p[20] = params['qgc'].value
    p[21] = params['qsoltr'].value
    p[22] = params['qinf'].value

    A_init = np.zeros((_0_config.state_num, _0_config.state_num))
    B_init = np.zeros((_0_config.state_num, _0_config.input_num))
    C_init = np.zeros((1, _0_config.state_num))
    D_init = np.zeros((1, _0_config.input_num))

    A, B, C, D = assgin_ABCD(A_init, B_init, C_init, D_init, p)

    sys = signal.StateSpace(A, B, C, D)
    sys_d = sys.to_discrete(_0_config.ts_sampling)
    a = sys_d.A
    b = sys_d.B
    c = sys_d.C
    d = sys_d.D

    return a, b, c, d


def resid(params, u_arr, y_arr):
    a, b, c, d = paras_to_ABCD(params)
    y_model = np.zeros_like(y_arr)
    # x_discrete = 25 * np.ones((7, u_arr.shape[1]))
    x_discrete = 25 * np.ones((7, 1))
    for i in range(u_arr.shape[1]):
        y_model[i,] = c @ x_discrete + d @ u_arr[:, i]
        x_discrete = a @ x_discrete + (b @ u_arr[:, i]).reshape((_0_config.state_num, 1))
    return y_model - y_arr


def load_init_rscs():
    with open('./init_rscs.csv', 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        for row in reader:
            rscs_str.append(row)
    rscs_str = rscs_str[0]
    rscs_lst = [float(s_num) for s_num in rscs_str]
    return rscs_lst


def init_para():
    rscs_init = load_init_rscs()
    params = lmfit.Parameters()
    params.add_many(('reout', rscs_init[0], 0.01*rscs_init[0], 100*rscs_init[0]),
                    ('re', rscs_init[1], 0.01*rscs_init[1], 100*rscs_init[1]),
                    ('rein', rscs_init[2], 0.01*rscs_init[2], 100*rscs_init[2]),
                    ('rcout', rscs_init[3], 0.01*rscs_init[3], 100*rscs_init[3]),
                    ('rc', rscs_init[4], 0.01*rscs_init[4], 100*rscs_init[4]),
                    ('rcin', rscs_init[5], 0.01*rscs_init[5], 100*rscs_init[5]),
                    ('rf', rscs_init[6], 0.01*rscs_init[6], 100*rscs_init[6]),
                    ('rfin', rscs_init[7], 0.01*rscs_init[7], 100*rscs_init[7]),
                    ('rw', rscs_init[8], 0.01*rscs_init[8], 100*rscs_init[8]),
                    ('ri1', rscs_init[9], 0.01*rscs_init[9], 100*rscs_init[9]),
                    ('ri2', rscs_init[10], 0.01*rscs_init[10], 100*rscs_init[10]),
                    ('ce1', rscs_init[11], 0.01*rscs_init[11], 100*rscs_init[11]),
                    ('ce2', rscs_init[12], 0.01*rscs_init[12], 100*rscs_init[12]),
                    ('cc1', rscs_init[13], 0.01*rscs_init[13], 100*rscs_init[13]),
                    ('cc2', rscs_init[14], 0.01*rscs_init[14], 100*rscs_init[14]),
                    ('cf', rscs_init[15], 0.01*rscs_init[15], 100*rscs_init[15]),
                    ('cair', rscs_init[16], 0.01*rscs_init[16], 100*rscs_init[16]),
                    ('ci1', rscs_init[17], 0.01*rscs_init[17], 100*rscs_init[17]),
                    ('ci2', rscs_init[18], 0.01*rscs_init[18], 100*rscs_init[18]),
                    ('qgc', rscs_init[19], 0.01*rscs_init[19],100*rscs_init[19]),
                    ('qsoltr', rscs_init[20], 0.01*rscs_init[20],100*rscs_init[20]),
                    ('qinf', rscs_init[21], 0.01*rscs_init[21],100*rscs_init[21]))
    return params


def assign_input_output(u_arr, y_arr, case_arr, ts):
    u_arr[:, 0] = case_arr[:, 0]
    u_arr[:, 1] = case_arr[:, 1]
    u_arr[:, 2] = case_arr[:, 7] + case_arr[:, 10] + case_arr[:, 13] + case_arr[:, 16]
    u_arr[:, 3] = case_arr[:, 6] + case_arr[:, 9] + case_arr[:, 12] + case_arr[:, 15]
    u_arr[:, 4] = case_arr[:, 22]
    u_arr[:, 5] = case_arr[:, 21]
    u_arr[:, 6] = case_arr[:, 18]
    u_arr[:, 7] = case_arr[:, 2]
    u_arr[:, 8] = case_arr[:, 3] + case_arr[:, 4]
    u_arr[:, 9] = -case_arr[:, 26] / ts
    u_arr[:, 10] = case_arr[:, 25]
    u_arr[0, 11] = 0
    u_arr[1:, 11] = (u_arr[1:, 10] - u_arr[0:-1, 10]) / ts

    y_arr[:, ] = (case_arr[:, 27] - case_arr[:, 28]) / ts

    return u_arr, y_arr


def load_u_y():
    # excluding the first row, column
    case_csv = pd.read_csv('./Case600.csv', index_col=0, parse_dates=True)
    case_arr = case_csv.to_numpy()[_0_config.start:_0_config.end]
    u_arr_init = np.zeros((case_arr.shape[0], _0_config.input_num))
    y_arr_init = np.zeros((case_arr.shape[0],))
    u_arr, y_arr = assign_input_output(u_arr_init, y_arr_init, case_arr, _0_config.ts_sampling)
    y_arr = pd.Series(y_arr)
    return (u_arr.T, y_arr)

def plot(o1, y_arr):
    max_heating = max(y_arr)
    min_heating = min(y_arr)
    # plt.plot([min_heating, max_heating], [min_heating, max_heating], '-', label='data')
    # plt.plot(y_arr, y_arr + o1.residual, 'o', label='modeled')
    plt.plot(y_arr, label='measured')
    plt.plot(y_arr + o1.residual, label='modeled')
    plt.legend()
    plt.show()

def swarm_plot(y_arr, y_arr_pred, title):
    plt.plot(y_arr, label='measured')
    plt.plot(y_arr_pred, label='modeled')
    plt.title(title)
    plt.legend()
    plt.show()
