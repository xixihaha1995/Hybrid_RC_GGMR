import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, json
from datetime import datetime

def add_internal(ambient_arr):
    # start =
    pass
    cur_utc = 1642204800+25200
    internal_heating = []
    for _ in range(len(ambient_arr)):
        isowkday = datetime.fromtimestamp(cur_utc).isoweekday()
        hour = datetime.fromtimestamp(cur_utc).hour
        if isowkday <= 5:
            if hour < 6:
                internal = 0
            elif hour >=6 and hour < 7:
                internal = 0.1
            elif hour >= 7 and hour < 8:
                internal = 0.2
            elif hour >= 8 and hour < 12:
                internal = 0.95
            elif hour >= 12 and hour < 13:
                internal = 0.5
            elif hour >= 13 and hour < 17:
                internal = 0.95
            elif hour >= 17 and hour < 18:
                internal = 0.3
            elif hour >= 18 and hour < 22 :
                internal = 0.1
            elif hour >= 22 and hour < 24:
                internal = 0.05
        if isowkday == 6:
            if hour < 6:
                internal = 0
            elif hour >= 6 and hour < 8:
                internal = 0.1
            elif hour >= 8 and hour < 12:
                internal = 0.3
            elif hour >= 12 and hour < 17:
                internal = 0.1
            elif hour >= 17 and hour < 19:
                internal = 0.05
            elif hour >= 19 and hour < 24:
                internal = 0
        if isowkday == 7:
            if hour < 6:
                internal = 0
            elif hour >= 6 and hour < 18:
                internal = 0.05
            elif hour >= 18 and hour < 24:
                internal = 0
        internal_heating.append([7.5* 30 *internal])
        cur_utc += 300

    ambient_arr = np.concatenate((ambient_arr, internal_heating), axis=1)
    return ambient_arr

def add_adj(ambient_arr):
    lst_2d = [[21] for _ in range(len(ambient_arr))]
    ambient_arr = np.concatenate((ambient_arr, np.array(lst_2d)), axis=1)
    return ambient_arr

def add_lighting(ambient_arr):
    # start =
    pass
    cur_utc = 1642204800+25200
    lighting = []
    for _ in range(len(ambient_arr)):
        isowkday = datetime.fromtimestamp(cur_utc).isoweekday()
        hour = datetime.fromtimestamp(cur_utc).hour
        if isowkday <= 5:
            if hour < 5:
                light = 0.05
            elif hour >=5 and hour < 7:
                light = 0.3
            elif hour >= 8 and hour < 12:
                light = 0.65
            elif hour >= 12 and hour < 13:
                light = 0.1
            elif hour >= 7 and hour < 8:
                light = 0.55
            elif hour >= 13 and hour < 17:
                light = 0.65
            elif hour >= 17 and hour < 18:
                light = 0.35
            elif hour >= 18 and hour < 20 :
                light = 0.3
            elif hour >= 20 and hour < 22:
                light = 0.2
            elif hour >= 22 and hour < 23:
                light = 0.1
            elif hour >= 23 and hour < 24:
                light = 0.05
        if isowkday == 6:
            if hour < 6:
                light = 0.05
            elif hour >= 6 and hour < 8:
                light = 0.1
            elif hour >= 8 and hour < 12:
                light = 0.3
            elif hour >= 12 and hour < 17:
                light = 0.15
            elif hour >= 17 and hour < 24:
                light = 0.05
        if isowkday == 7:
            light = 0.05
        lighting.append([7.5* 30 *light])
        cur_utc += 300

    ambient_arr = np.concatenate((ambient_arr, lighting), axis=1)
    return ambient_arr



def loadJSON(name):
    with open(os.path.join(name + '.json'), 'r') as f:
        testDict = json.loads(f.read())
    return testDict


def saveJSON(data, name):
    with open(os.path.join(name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_u_y(constants, train=True):
    if constants['case_nbr'] == -1:
        case_csv = pd.read_csv('./Case600.csv', index_col=0, parse_dates=True)
        if not train:
            case_arr = case_csv.to_numpy()[constants['start'] + constants['end']:constants['end'] * 2]
        else:
            case_arr = case_csv.to_numpy()[constants['start']: constants['end']]
    else:
        case_csv = pd.read_csv('./RS_VAV_baseline_1_15_3_7.csv', index_col=0, parse_dates=True,
                               encoding='unicode_escape')
        ambient_csv = pd.read_csv('./ambient-weather-20220115-20220307_2.csv', index_col=0, parse_dates=True)
        ambient_arr = add_internal(ambient_csv.to_numpy())
        ambient_arr = add_adj(ambient_arr)
        ambient_arr = add_lighting(ambient_arr)
        if not train:
            case_arr = case_csv.to_numpy()[constants['start'] + constants['end']:constants['end'] * 2]
            ambient_arr = ambient_arr[constants['start'] + constants['end']:constants['end'] * 2]
        else:
            case_arr = case_csv.to_numpy()[constants['start']: constants['end']]
            ambient_arr = ambient_arr[constants['start']: constants['end']]
        case_arr = np.concatenate((case_arr, ambient_arr), axis=1)

    u_arr_init = np.zeros((case_arr.shape[0], constants['input_num']))
    y_arr_init = np.zeros((case_arr.shape[0],))
    u_arr, y_arr = assign_input_output(u_arr_init, y_arr_init, case_arr, constants['ts_sampling'],
                                       case_nbr=constants['case_nbr'])
    y_arr = pd.Series(y_arr)
    return (u_arr.T, y_arr)


def assign_input_output(u_arr, y_arr, case_arr, ts, case_nbr=3):
    return_temp_c = (case_arr[:, 29] - 32) * 5 / 9
    sulp_temp_c = (case_arr[:, 28] - 32) * 5 / 9
    flow_volume_rate_gal_min = case_arr[:, 27]
    # c = 4.186 J/g/c, rho = 997e3 g/m3, 1 gal / min = 6.309e-5 m3/s,
    c = 4.186
    rho = 997e3
    gal_permin_to_m3_persecond = 6.309e-5
    t_slab = ((case_arr[:, 5] + case_arr[:, 6] + case_arr[:, 7] + case_arr[:, 8] + case_arr[:, 9]
               + case_arr[:, 10] + case_arr[:, 11] + case_arr[:, 12] + case_arr[:, 13] + case_arr[:,
                                                                                         14]) / 10 - 32) * 5 / 9
    ligthing_power =case_arr[:, -1]
    # c_air = 700 J/kg/K, 0.00047194745 m3_per3_perCFM, rho_air = 1.255 kg/m3
    c_air = 700
    rho_air = 1.255
    out_temp = (case_arr[:, 38] - 32)*5 / 9
    cfm_1 = case_arr[:, 41]
    t_supp_1 = (case_arr[:, 42] - 32 ) * 5/ 9
    cfm_2 = case_arr[:, 45]
    t_supp_2 = (case_arr[:, 46] - 32) * 5 / 9
    m3_per3_perCFM = 0.00047194745
    # QAHU = c_air*cfm * m3_per3_perCFM *rho_air*(T_supp - OA)

    if case_nbr == -1:
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
    elif case_nbr == 0:
        #     ut = t room, t out,  q sol
        #     yt = t cav
        u_arr[:, 0] = (case_arr[:, 2] - 32) * 5 / 9
        u_arr[:, 1] = (case_arr[:, 0] - 32) * 5 / 9
        u_arr[:, 2] = case_arr[:, 57 + 17]
        y_arr[:, ] = (case_arr[:, 49] - 32) * 5 / 9
    elif case_nbr == 1:
        #     ut = t out, Tcav Tslab, Qsol, Qlight, Qint, Qahu,
        #     yt = t room
        # QAHU = c_air*cfm * m3_per3_perCFM *rho_air*(T_supp - OA)
        u_arr[:, 0] = (case_arr[:, 0] - 32) * 5 / 9
        u_arr[:, 1] = (case_arr[:, 49] - 32) * 5 / 9
        u_arr[:, 2] = t_slab
        u_arr[:, 3] = case_arr[:, 57 + 17]
        u_arr[:, 4] = ligthing_power
        u_arr[:, 5] = case_arr[:,-3]
        u_arr[:, 6] = c_air * m3_per3_perCFM * rho_air * (cfm_1 * (t_supp_1 - out_temp) + cfm_2 * (t_supp_2 - out_temp))

        y_arr = (case_arr[:, 2] - 32) * 5 / 9


    elif case_nbr == 2:
        #     ut = t room, q sol, q light, dT_{source} / dt
        #     yt = q rad
        pass
        u_arr[:, 0] = (case_arr[:, 2] - 32) * 5 / 9
        u_arr[:, 1] = case_arr[:, 57 + 17]
        u_arr[:, 2] = ligthing_power
        t_so = (sulp_temp_c + return_temp_c) / 2
        # t_so = sulp_temp_c
        u_arr[:-1, 3] = (t_so[1:] - t_so[:-1] ) /ts
        u_arr[-1, 3] = 0

        y_arr[:, ] = c * rho * flow_volume_rate_gal_min * gal_permin_to_m3_persecond * (sulp_temp_c - return_temp_c)

    elif case_nbr == 3:
        # uT = [T_{out}, \dot{Q}_{sol, cav}, \dot{Q}_{sol, room}, \dot{Q}_{int, room}, \dot{Q}_{sol, sur}, \dot{Q}_{int, sur}, \frac{dT_{so}}{dt}]\\
        u_arr[:, 0] = (case_arr[:, 0] - 32) * 5 / 9
        radiation = case_arr[:, 57 + 17]
        u_arr[:, 1] = radiation * 0.04
        u_arr[:, 2] = radiation * 3.7e-19 * 0.5
        u_arr[:, 3] = np.zeros_like(u_arr[:, 0])
        u_arr[:, 4] = radiation * 3.7e-19 * 0.5
        u_arr[:, 5] = np.zeros_like(u_arr[:, 0])

        u_arr[:, 6] = ((case_arr[:, 29] + case_arr[:, 28]) / 2 - 32) * 5 / 9
        u_arr[0, 6] = 0
        u_arr[1:, 6] = (u_arr[1:, 6] - u_arr[0:-1, 6]) / ts

        delta_t = (return_temp_c - sulp_temp_c)
        y_arr[:, ] = c * rho * flow_volume_rate_gal_min * gal_permin_to_m3_persecond * delta_t

    elif case_nbr == 4:
        #     ut = t room, q sol,  q rad, q light, t adj
        #     yt = t slab
        pass
        u_arr[:, 0] = (case_arr[:, 2] - 32) * 5 / 9
        u_arr[:, 1] = case_arr[:, 57 + 17]
        u_arr[:, 2] = c * rho * flow_volume_rate_gal_min * gal_permin_to_m3_persecond * (sulp_temp_c - return_temp_c)
        u_arr[:, 3] = ligthing_power
        u_arr[:, 4] = case_arr[:, -2]

        y_arr[:, ] = t_slab

    return u_arr, y_arr


def assgin_ABCD(A, B, C, D, p, case_nbr=3):
    if case_nbr == -1:
        A[1 - 1, 1 - 1] = -(1 / (p[0] * p[11])) - (1 / (p[1] * p[11]))
        A[1 - 1, 2 - 1] = (1 / (p[1] * p[11]))
        A[2 - 1, 1 - 1] = 1 / (p[1] * p[12])
        A[2 - 1, 2 - 1] = -(1 / (p[1] * p[12]) + 1 / (p[2] * p[12]))
        A[3 - 1, 3 - 1] = -(1 / (p[3] * p[13]) + 1 / (p[4] * p[13]))
        A[3 - 1, 4 - 1] = 1 / (p[4] * p[13])
        A[4 - 1, 3 - 1] = 1 / (p[4] * p[14])
        A[4 - 1, 4 - 1] = -(1 / (p[4] * p[14]) + 1 / (p[5] * p[14]))
        A[5 - 1, 5 - 1] = -(1 / (p[6] * p[15]) + 1 / (p[7] * p[15]))
        A[6 - 1, 6 - 1] = -1 / (p[9] * p[17])
        A[6 - 1, 7 - 1] = 1 / (p[9] * p[17])
        A[7 - 1, 6 - 1] = 1 / (p[9] * p[18])
        A[7 - 1, 7 - 1] = -1 / (p[9] * p[18]) - 1 / (p[10] * p[18])

        B[1 - 1, 1 - 1] = (1 / (p[0] * p[11]))
        B[1 - 1, 3 - 1] = 1 / p[11]
        B[2 - 1, 4 - 1] = 1 / p[12]
        B[2 - 1, 11 - 1] = 1 / (p[2] * p[12])
        B[3 - 1, 1 - 1] = 1 / (p[3] * p[13])
        B[3 - 1, 5 - 1] = 1 / p[13]
        B[4 - 1, 6 - 1] = 1 / p[14]
        B[4 - 1, 11 - 1] = 1 / (p[5] * p[14])
        B[5 - 1, 2 - 1] = 1 / (p[6] * p[15])
        B[5 - 1, 7 - 1] = 1 / (p[15])
        B[5 - 1, 11 - 1] = 1 / (p[7] * p[15])
        B[6 - 1, 9 - 1] = p[20] / (2 * p[17])
        B[7 - 1, 9 - 1] = p[20] / (2 * p[18])
        B[7 - 1, 11 - 1] = 1 / (p[10] * p[18])

        C[1 - 1, 2 - 1] = -1 / p[2]
        C[1 - 1, 4 - 1] = -1 / p[5]
        C[1 - 1, 5 - 1] = -1 / p[7]
        C[1 - 1, 7 - 1] = -1 / p[10]

        D[1 - 1, 1 - 1] = -1 / p[8]
        D[1 - 1, 8 - 1] = -p[19]
        D[1 - 1, 10 - 1] = -p[21]
        D[1 - 1, 11 - 1] = 1 / p[2] + 1 / p[5] + 1 / p[7] + 1 / p[10] + 1 / p[8]
        D[1 - 1, 12 - 1] = p[16]

    elif case_nbr == 0:
        pass
        A[0, 0] = -(1 / (p[0] * p[2])) - (1 / (p[1] * p[2]))
        B[0, 0] = 1 / (p[1] * p[2])
        B[0, 1] = 1 / (p[0] * p[2])
        B[0, 2] = p[3] / p[2]

        C[0, 0] = 1
    elif case_nbr == 1:
        pass
        A[0, 0] = -1 / (p[0] * p[4]) + -1 / (p[1] * p[4])
        A[0, 1] = 1 / (p[1] * p[4])
        A[1, 0] = 1 / (p[1] * p[5])
        A[1, 1] = -1 / (p[1] * p[5])

        B[0, 1] = 1 /(p[0] * p[4])
        B[0, 3] = p[6] / p[4]
        B[1, 1] = 1 / (p[2] * p[5])
        B[1, 2] = 1 / (p[3] * p[5])
        B[1, 3] = p[7] / p[5]
        B[1, 4] = p[8] / p[5]
        B[1, 5] = p[9] / p[5]
        B[1, 6] = 1

        C[0, 1] = p[10]

    elif case_nbr == 2:
        pass
        A[0, 0] = -1 / (p[0] * p[3]) + -1 / (p[1] * p[3])
        A[0, 1] = 1 / (p[1] * p[3])
        A[1, 0] = 1 / (p[1] * p[4])
        A[1, 1] = -1 / (p[1] * p[4]) + -1 / (p[2] * p[4])
        A[1, 2] = 1 / (p[2] * p[4])
        A[2, 1] = 1 / (p[2] * p[5])
        A[2, 2] = -1 / (p[2] * p[5])

        B[0, 0] = 1 / (p[0] * p[3])
        B[0, 1] = p[6] / p[3]
        B[0, 2] = p[7] / p[3]

        C[0, 0] = -1 / p[1]
        C[0, 1] = (1 / p[1] + 1 / p[2])
        C[0, 2] = -1 / p[2]

        D[0, 3] = p[4]


    elif case_nbr == 3:
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

    elif case_nbr == 4:
        pass
        A[0, 0] = -1 / (p[0] * p[3]) + -1 / (p[1] * p[3])
        A[0, 1] = 1 / (p[1] * p[3])
        A[1, 0] = 1 / (p[1] * p[4])
        A[1, 1] = -1 / (p[1] * p[4]) + -1 / (p[2] * p[4])


        B[0, 0] = 1 / (p[0] * p[3])
        B[0, 1] = p[5] / p[3]
        B[0, 3] = p[6] / p[3]
        B[1, 2] = 1 / p[4]
        B[1, 4] = 1 /(p[2] * p[4])

        C[0, 0] = 1
    return A, B, C, D


def nrmse(measure, model):
    nom = (sum((measure - model) ** 2) / len(measure)) ** 1 / 2
    if not isinstance(nom, float):
        nom = nom[0,0]
    # mean = measure.mean()
    # denom = (sum((measure - mean) ** 2)) ** 1 / 2
    # if denom == 0:
    #     denom = len(measure)
    return nom


def swarm_plot(y_train, y_train_pred, y_test, y_test_pred, swarm_constants):
    fig, ax = plt.subplots(2)
    nl = '\n'
    minutes_interval = swarm_constants['ts_sampling'] / 60
    start = swarm_constants['start']
    end = swarm_constants['end']
    if swarm_constants['case_nbr'] == -1:
        figure_title = f'Single Zone RC network for Heating Power prediction(J){nl}'
    elif swarm_constants['case_nbr'] == 0:
        figure_title = f'Cav RC network for T_cav prediction(C){nl}'
    elif swarm_constants['case_nbr'] == 1:
        figure_title = f'Room RC network for T_room prediction(C){nl}'
    elif swarm_constants['case_nbr'] == 2:
        figure_title = f'Slab RC network for Q_rad prediction(W){nl}'
    elif swarm_constants['case_nbr'] == 3:
        figure_title = f'Integrated RC network for Heating power(J) prediction performance{nl}'
    elif swarm_constants['case_nbr'] == 4:
        figure_title = f'Slab RC network (Sink is temperature boundary as 21 C) for T_slab prediction(C){nl}'
    ax[0].plot(y_train, label='measured')
    ax[0].plot(y_train_pred, label='modeled')
    ax[0].set_title(
        f'Train, from {start * minutes_interval}th mins to {end * minutes_interval}th mins, RMSE:{nrmse(y_train, y_train_pred):.2f}')
    if swarm_constants['case_nbr'] == 2:
        ax[0].set_ylim((None, None))
    ax[1].plot(y_test, label='measured')
    ax[1].plot(y_test_pred, label='modeled')
    ax[1].set_title(
        figure_title + nl + f'Test, from {(start + end) * minutes_interval}th mins to {end * 2 * minutes_interval}th mins, RMSE:{nrmse(y_test, y_test_pred):.2f}')

    if swarm_constants['case_nbr'] == 2:
        ax[1].set_ylim((None, None))
    plt.legend()
    plt.subplots_adjust(hspace=0.8)
    case_nbr = swarm_constants['case_nbr']
    plt.savefig(f'_{case_nbr}_swarm_performance.png')
    plt.show()


def pos_subplot(data, title, ax0=None, ax1=None):
    nl = '\n'
    if not ax1:
        ax = plt.gca()
    ax0.plot(data[0], label='train measured')
    ax0.plot(data[1], label='train modeled')
    ax0.set_title(title + f'{nl}RMSE:{nrmse(data[0], data[1]):.6f}')
    ax0.legend()

    ax1.plot(data[2], label='test measured')
    ax1.plot(data[3], label='test modeled')
    ax1.set_title(f'RMSE:{nrmse(data[2], data[3]):.6f}')
    ax1.legend()


def pos_plot_all(all_pos):
    f, ax = plt.subplots(len(all_pos), 2)
    for i in range(len(all_pos)):
        pos_subplot(all_pos[i]['performance'], all_pos[i]['title'], ax[i][0], ax[i][1])
    plt.legend()
    plt.subplots_adjust(hspace=0.8)
    plt.show()
