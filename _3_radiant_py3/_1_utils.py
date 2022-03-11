import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def load_u_y(constants, train=True):
    # excluding the first row, column

    case_csv = pd.read_csv('./RS_baseline_1_15_3_7.csv', index_col=0, parse_dates=True)
    ambient_csv = pd.read_csv('./ambient-weather-20220115-20220307_2.csv', index_col=0, parse_dates=True)
    if not train:
        case_arr = case_csv.to_numpy()[constants['start'] + constants['end']:constants['end'] * 2]
        ambient_arr = ambient_csv.to_numpy()[constants['start'] + constants['end']:constants['end'] * 2]
    else:
        case_arr = case_csv.to_numpy()[constants['start'] : constants['end']]
        ambient_arr = ambient_csv.to_numpy()[constants['start'] : constants['end']]
    case_arr = np.concatenate((case_arr, ambient_arr), axis = 1)
    # radiation should be column 45 + 17 = 62

    # np.concatenate((a, b))
    u_arr_init = np.zeros((case_arr.shape[0], constants['input_num']))
    y_arr_init = np.zeros((case_arr.shape[0],))
    u_arr, y_arr = assign_input_output(u_arr_init, y_arr_init, case_arr,  constants['ts_sampling'])
    y_arr = pd.Series(y_arr)
    return (u_arr.T, y_arr)


def assign_input_output(u_arr, y_arr, case_arr, ts):
    # uT = [T_{out}, \dot{Q}_{sol, cav}, \dot{Q}_{sol, room}, \dot{Q}_{int, room}, \dot{Q}_{sol, sur}, \dot{Q}_{int, sur}, \frac{dT_{so}}{dt}]\\

    u_arr[:, 0] = (case_arr[:, 0] - 32) * 5 / 9

    radiation = case_arr[:, 62]
    u_arr[:, 1] = radiation * 0.04
    u_arr[:, 2] = radiation * 3.7e-19 * 0.5
    u_arr[:, 3] = np.zeros_like(u_arr[:, 0])
    u_arr[:, 4] = radiation * 3.7e-19 * 0.5
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
    # 0. r out cav, 0.036 K/W, corrected by Jaewan 1 / 51.92 = 0.019
    # 1. r cav room 0.0036 K/W, corrected by Jaewan 1 / 29.07 = 0.034
    # 2. r out room 0.036 K/W wood door, corrected by Jaewan 0.046
    # 3. r room sur 10 K/W, indoor heat transfer coefficient 10 - 500 W/(m2 K), corrected by Jaewan 1 / 527.58 = 1.89E-3
    # 4. r sur so 40 K/W 4 inch concrete, 100 m2 * 0.4 m2xK/W,  corrected by Jaewan 1 / 1402.44 = 7E-4
    # 5. r si so 300 common thermal insulation r value = 3 m2xK/W, corrected by Jaewan 1 / 249.39 = 4e-3
    # 6. c cav 75300 J/K, corrected by Jaewan 1 / 1.13e-6 = 884955
    # 7  c room 376500 J/K, corrected by Jaewan 1 / 2.43e-7 = 4115226
    # 8  c sur, concrete 0.88 J/g/K, 2.4e6 g /m3 0.1 * 10 * 10 m3 = 2e7, corrected by Jaewan 1 / 3.53e-8 = 2.8E7
    # (c = 4.186 J / g / c, rho = 997e3 g / m3, 1 gal / min = 6.309e-5 m3 / s,
    # 9  c so, 4.18 J / g / c  * water 10 gal/min *  6.309e-5 m3 / s * 997e3 g / m3 = 2629 J/K, corrected by Jaewan, 1 / 3.61e-7 = 2.7E6
    # 10 c si, 2100 J/(kg K) * 160 kg /m3 * 0.1 * 10 * 10 m3 = 3360000, corrected by Jaewan 1 / 4.14e-17 = 2.415E16
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


def swarm_plot(y_train, y_train_pred, y_test, y_test_pred, swarm_constants):
    fig, ax = plt.subplots(2)
    nl = '\n'
    minutes_interval = swarm_constants['ts_sampling'] / 60
    start = swarm_constants['start']
    end = swarm_constants['end']

    figure_title = f'Heating power(J) prediction performance{nl} with Particle Swarm Optimization (PSO)'
    # plt.suptitle(figure_title)
    ax[0].plot(y_train, label='measured')
    ax[0].plot(y_train_pred, label='modeled')
    ax[0].set_title(
        f'Train, from {start * minutes_interval}th mins to {end * minutes_interval}th mins, NRMSE:{nrmse(y_train, y_train_pred):.2f}')

    ax[1].plot(y_test, label='measured')
    ax[1].plot(y_test_pred, label='modeled')
    ax[1].set_title(
        figure_title + nl + f'Test, from {(start +end) * minutes_interval}th mins to {end * 2 * minutes_interval}th mins, NRMSE:{nrmse(y_test, y_test_pred):.2f}')

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
