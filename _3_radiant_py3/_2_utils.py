import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _1_config


def load_u_y(train=True):
    # excluding the first row, column
    case_csv = pd.read_csv('./RS_baseline_1_15_3_7.csv', index_col=0, parse_dates=True)
    if not train:
        case_arr = case_csv.to_numpy()[_1_config.start + _1_config.end:_1_config.end * 2]
    else:
        case_arr = case_csv.to_numpy()[_1_config.start:_1_config.end]
    u_arr_init = np.zeros((case_arr.shape[0], _1_config.input_num))
    y_arr_init = np.zeros((case_arr.shape[0],))
    u_arr, y_arr = assign_input_output(u_arr_init, y_arr_init, case_arr, _1_config.ts_sampling)
    y_arr = pd.Series(y_arr)
    return (u_arr.T, y_arr)


def assign_input_output(u_arr, y_arr, case_arr, ts):
    # uT = [T_{out}, \dot{Q}_{sol, cav}, \dot{Q}_{sol, room}, \dot{Q}_{int, room}, \dot{Q}_{sol, sur}, \dot{Q}_{int, sur}, \frac{dT_{so}}{dt}]\\
    u_arr[:, 0] = case_arr[:, 0]
    u_arr[:, 1] = np.zeros_like(u_arr[:, 0])
    u_arr[:, 2] = np.zeros_like(u_arr[:, 0])
    u_arr[:, 3] = np.zeros_like(u_arr[:, 0])
    u_arr[:, 4] = np.zeros_like(u_arr[:, 0])
    u_arr[:, 5] = np.zeros_like(u_arr[:, 0])

    u_arr[:, 6] = ((case_arr[:, 41] + case_arr[:, 40]) / 2 - 32) * 5 / 9
    u_arr[0, 6] = 0
    u_arr[1:, 6] = (u_arr[1:, 6] - u_arr[0:-1, 6]) / ts

    return_temp_c = (case_arr[:, 41] - 32) * 5 / 9
    sulp_temp_c = (case_arr[:, 40] - 32) * 5 / 9
    flow_volume_rate_gal_min = case_arr[:, 39]
    # c = 4.186 J/g/c, rho = 997e3 g/m3, 1 gal / min = 6.309e-5 m3/s,
    c = 4.186
    rho = 997e3
    gal_permin_to_m3_persecond = 6.309e-5
    delta_t = (return_temp_c - sulp_temp_c)

    y_arr[:, ] = c * rho * flow_volume_rate_gal_min * gal_permin_to_m3_persecond * delta_t

    return u_arr, y_arr


def assgin_ABCD(A, B, C, D, p):
    # 0. r out cav
    # 1. r cav room
    # 2. r out room
    # 3. r room sur
    # 4. r sur so
    # 5. r si so
    # 6. c cav
    # 7  c room
    # 8  c sur
    # 9  c so
    # 10 c si
    A[0, 0] = -(1 / (p[0] * p[6])) - (1 / (p[1] * p[6]))
    A[0, 1] = (1 / (p[1] * p[6]))

    A[1, 0] = 1 / (p[1] * p[7])
    A[1, 1] = -1 / (p[1] * p[7]) - 1 / (p[3] * p[7]) - 1 / (p[1] * p[7])
    A[1, 2] = 1 / (p[3] * p[7])

    A[2, 1] = 1 / (p[3] * p[8])
    A[2, 2] = -1 / (p[3] * p[8]) - 1 / (p[4] * p[8])
    A[2, 3] = 1 / (p[4] * p[8])

    A[3, 2] = 1 / (p[4] * p[9])
    A[3, 3] = -1 / (p[4] * p[9]) - 1 / (p[5] * p[9])
    A[3, 4] = 1 / (p[5] * p[9])

    A[4, 3] = 1 / (p[5] * p[10])
    A[4, 4] = -1 / (p[5] * p[10])

    B[0, 0] = 1 / (p[0] * p[6])
    B[0, 1] = 1

    B[1, 0] = 1 / (p[2] * p[7])
    B[1, 2] = 1
    B[1, 3] = 1

    B[2, 4] = 1
    B[2, 5] = 1

    C[0, 2] = 1 / p[4]
    C[0, 3] = -1 / p[4] - 1 / p[5]

    C[0, 4] = 1 / p[5]

    D[0, 6] = -p[9]

    return A, B, C, D


def nrmse(measure, model):
    nom = (sum((measure - model) ** 2)) ** 1 / 2
    mean = measure.mean()
    denom = (sum((measure - mean) ** 2)) ** 1 / 2
    return nom / denom


def swarm_plot(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(2)
    nl = '\n'
    minutes_interval = _1_config.ts_sampling / 60
    figure_title = f'Heating power(J) prediction performance{nl} with Particle Swarm Optimization (PSO)'
    # plt.suptitle(figure_title)
    ax[0].plot(y_train, label='measured')
    ax[0].plot(y_train_pred, label='modeled')
    ax[0].set_title(
        f'Train, from {_1_config.start * minutes_interval}th mins to {_1_config.end * minutes_interval}th mins, NRMSE:{nrmse(y_train, y_train_pred):.2f}')

    ax[1].plot(y_test, label='measured')
    ax[1].plot(y_test_pred, label='modeled')
    ax[1].set_title(
        figure_title + nl + f'Test, from {(_1_config.start + _1_config.end) * minutes_interval}th mins to {_1_config.end * 2 * minutes_interval}th mins, NRMSE:{nrmse(y_test, y_test_pred):.2f}')

    plt.legend()
    plt.subplots_adjust(hspace=0.8)
    plt.savefig("swarm_performance.png")
    plt.show()


def pos_subplot(data, title, ax0=None, ax1=None):
    nl = '\n'
    if not ax1:
        ax = plt.gca()
    ax0.plot(data[0], label='train measured')
    ax0.plot(data[1], label='train modeled')
    ax0.set_title(title + f'{nl}NRMSE:{nrmse(data[0], data[1]):.6f}')
    ax0.legend()

    ax1.plot(data[2], label='test measured')
    ax1.plot(data[3], label='test modeled')
    ax1.set_title(f'NRMSE:{nrmse(data[2], data[3]):.6f}')
    ax1.legend()


def pos_plot_all(all_pos):
    f, ax = plt.subplots(len(all_pos), 2)
    for i in range(len(all_pos)):
        pos_subplot(all_pos[i]['performance'], all_pos[i]['title'], ax[i][0], ax[i][1])
    plt.legend()
    plt.subplots_adjust(hspace=0.8)
    plt.show()
