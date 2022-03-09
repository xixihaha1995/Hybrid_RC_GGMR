import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _1_config

def load_u_y(train = True):
    # excluding the first row, column
    case_csv = pd.read_csv('./Case600.csv', index_col=0, parse_dates=True)
    if not train :
        case_arr = case_csv.to_numpy()[_1_config.start + _1_config.end:_1_config.end * 2]
    else:
        case_arr = case_csv.to_numpy()[_1_config.start:_1_config.end]
    u_arr_init = np.zeros((case_arr.shape[0], _1_config.input_num))
    y_arr_init = np.zeros((case_arr.shape[0],))
    u_arr, y_arr = assign_input_output(u_arr_init, y_arr_init, case_arr, _1_config.ts_sampling)
    y_arr = pd.Series(y_arr)
    return (u_arr.T, y_arr)

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

def nrmse(measure, model):
    nom = (sum((measure - model) ** 2)) ** 1 / 2
    mean = measure.mean()
    denom = (sum((measure - mean) ** 2)) ** 1 / 2
    return nom / denom


def swarm_plot(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(2)
    nl = '\n'
    figure_title = f'Heating power(J) prediction performance{nl} with Particle Swarm Optimization (PSO)'
    # plt.suptitle(figure_title)
    ax[0].plot(y_train, label='measured')
    ax[0].plot(y_train_pred, label='modeled')
    ax[0].set_title(
        f'Train, from {_1_config.start * 2}th mins to {_1_config.end * 2}th mins, NRMSE:{nrmse(y_train, y_train_pred):.2f}')

    ax[1].plot(y_test, label='measured')
    ax[1].plot(y_test_pred, label='modeled')
    ax[1].set_title(
        figure_title + nl + f'Test, from {(_1_config.start + _1_config.end) * 2}th mins to {_1_config.end * 2 * 2}th mins, NRMSE:{nrmse(y_test, y_test_pred):.2f}')

    plt.legend()
    plt.subplots_adjust(hspace=0.8)
    plt.savefig("swarm_performance.png")
    plt.show()


def pos_subplot(data, title, ax0=None, ax1 = None):
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
    f, ax = plt.subplots(len(all_pos),2)
    for i in range(len(all_pos)):
        pos_subplot(all_pos[i]['performance'], all_pos[i]['title'], ax[i][0], ax[i][1])
    plt.legend()
    plt.subplots_adjust(hspace=0.8)
    plt.show()
