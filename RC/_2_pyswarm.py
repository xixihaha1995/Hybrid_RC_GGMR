import numpy as np, pandas as pd
from . import _1_utils
import csv, os, random
from scipy import signal
import matplotlib.pyplot as plt

u_train = None
y_train = None
u_test = None
y_test = None
load_u_y_bool = False

load_all_case_arr = False
all_case_arr_arr =None

def _statistical_distribution_best_warming_up():
    pos = [0.31104853199463756, 0.5467046592684783, 0.9935975211504724, -0.014585286370612916, 0.721647509968919,
           0.4686516990698716, 0.000565144547088079, 0.0006441690230600109, 0.0009074541207515334, 2600000.379360871,
           1300000.4824038981, 100000000.82953805, 1200000.0401698523, 5999999.992084267, 19999.996035939595,
           274999.9999740066, 100.17175908793585, 0.1872934280213577, 1.5246900378107624, 2.460589040747263,
           1.4087593540485726, 0.4659914186864218, 1.013398093721322, 1.7993314767358861]

    tim_start_arr = [random.randint(100, 14000) for i in range(100)]
    segment_len = range(1, 41)

    all_optimized_warming = []
    for time_start in tim_start_arr:
        predicted_err = []
        for segment in segment_len:
            cur_case_arr = different_warming_up(time_idx=time_start, seg_length=segment)
            (u_measured_arr, y_measured_arr) = warming_up_init_assign_u_y(cur_case_arr)
            y_measured_arr = y_measured_arr.to_numpy()
            y_model_arr = warming_up_predict(pos, u_measured_arr)
            predicted_err.append(abs(y_model_arr[-1] - y_measured_arr[-1]) / abs(y_measured_arr[-1]))
        all_optimized_warming.append(predicted_err)
    plt.plot(np.mean(all_optimized_warming, axis=0))
    plt.ylabel("Mean Absolute Percentage Error")
    plt.xlabel("Warming time steps (5 mins per step)")
    plt.show()

def warming_input_demo(time_idx, seg_length):
    case_arr_cur = different_warming_up(time_idx, seg_length)
    (u_arr_Tran, y_arr) = warming_up_init_assign_u_y(case_arr_cur)
    return u_arr_Tran


def different_warming_up(time_idx, seg_length):
    global load_all_case_arr, all_case_arr_arr
    if not load_all_case_arr:
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        all_case_arr_abs = os.path.join(script_dir, 'inputs', 'case_arr.csv')
        all_case_arr_arr = pd.read_csv(all_case_arr_abs).to_numpy()
        load_all_case_arr = True

    sliced_case_arr = all_case_arr_arr[time_idx - seg_length + 1: time_idx + 1]
    return sliced_case_arr

def warming_up_init_assign_u_y(case_arr, _case_nbr=6, _ts=300, _input_num=9):
    u_arr_init = np.zeros((case_arr.shape[0], _input_num))
    y_arr_init = np.zeros((case_arr.shape[0],))
    u_arr, y_arr = _1_utils.assign_input_output(u_arr_init, y_arr_init, case_arr, _ts,
                                       case_nbr=_case_nbr)
    y_arr = pd.Series(y_arr)
    return (u_arr.T, y_arr)

def warming_up_predict(u_arr, pos = None,  _case_nbr=6, _ts=300, _state_num=6, _input_num=9):
    if not pos:
        pos = [0.31104853199463756, 0.5467046592684783, 0.9935975211504724, -0.014585286370612916, 0.721647509968919,
               0.4686516990698716, 0.000565144547088079, 0.0006441690230600109, 0.0009074541207515334, 2600000.379360871,
               1300000.4824038981, 100000000.82953805, 1200000.0401698523, 5999999.992084267, 19999.996035939595,
               274999.9999740066, 100.17175908793585, 0.1872934280213577, 1.5246900378107624, 2.460589040747263,
               1.4087593540485726, 0.4659914186864218, 1.013398093721322, 1.7993314767358861]

    constants = {}
    constants['state_num'] = _state_num
    constants['input_num'] = _input_num
    constants['case_nbr'] = _case_nbr
    constants['ts_sampling'] = _ts

    a, b, c, d = paras_to_ABCD_swarm(pos, constants)
    y_model = np.zeros((u_arr.shape[1],))

    x_discrete = np.array([[0], [10],[22],[21],[23],[21]])

    for i in range(u_arr.shape[1]):
        y_model[i] = (c @ x_discrete + d @ u_arr[:, i])[0,0]
        x_discrete = a @ x_discrete + (b @ u_arr[:, i]).reshape((_state_num, 1))

    return y_model

def plot_state_variables_dynamis(x_all):
    plt.plot(x_all[:,0], label = "Envelop 1")
    plt.plot(x_all[:,1], label="Envelop 2")
    plt.plot(x_all[:,2], label="Room")
    plt.plot(x_all[:,3], label="Internal Wall")
    plt.plot(x_all[:,4], label="Slab")
    plt.plot(x_all[:,5], label="Sink")
    plt.ylim((-10, 100))
    plt.xlabel("Time steps, interval = 5 mins")
    plt.ylabel("Temperature degree Celsius")
    plt.title("State Variable dynamics from Jan 15, 2022 to March 7, 2022 for Radiant Slab System RC")
    plt.legend()
    plt.show()


def init_pos(case_nbr, n_particles):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'inputs','init_rscs.txt'), 'r') as f:
        reader = csv.reader(f)
        rscs_str = []
        for row in reader:
            row = [float(s_num) for s_num in row]
            rscs_str.append(row)
    rscs_lst = rscs_str[case_nbr]
    rscs_init = np.array([rscs_lst for _ in range(n_particles)])
    return rscs_init


def whole_swarm_loss(x, constants):
    n_particles = x.shape[0]
    j = [particle_loss(x[i], constants) for i in range(n_particles)]
    return np.array(j).reshape(n_particles)


def particle_loss(params, constants):
    y_measure, y_model = obj_func(params, constants)
    y_measure = y_measure.to_numpy()
    return sum((y_model - y_measure) ** 2)


def obj_func(params, constants, train=True):
    if not load_u_y_bool:
        call_load_u_y(constants)
    if train:
        u_arr, y_arr = u_train, y_train
    else:
        u_arr, y_arr = u_test, y_test
    a, b, c, d = paras_to_ABCD_swarm(params, constants)
    y_model = np.zeros_like(y_arr)

    if constants['case_nbr'] == -1:
        x_discrete = 25 * np.ones((constants['state_num'], 1))
    elif constants['case_nbr'] == 0:
        x_discrete = np.array([[22]])
    elif constants['case_nbr'] == 1:
        x_discrete = np.array([[0],[22]])
    elif constants['case_nbr'] == 2:
        x_discrete = np.array([[22], [30], [21]])
    elif constants['case_nbr'] == 3:
        x_discrete = np.array([[11], [22], [22], [25], [25]])
    elif constants['case_nbr'] == 4:
        x_discrete = np.array([[22], [27]])
    elif constants['case_nbr'] == 5:
        x_discrete = np.array([[0], [10],[22],[21]])
    elif constants['case_nbr'] == 6:
        x_discrete = np.array([[0], [10],[22],[21],[23],[21]])
    state_num = constants['state_num']

    if constants['inspect_x_state'] and not train:
        x_all=[]
    for i in range(u_arr.shape[1]):
        y_model[i] = (c @ x_discrete + d @ u_arr[:, i])[0,0]
        x_discrete = a @ x_discrete + (b @ u_arr[:, i]).reshape((state_num, 1))
        if constants['inspect_x_state'] and not train:
            x_all.append(x_discrete.reshape(constants['state_num'],))


    if constants['inspect_x_state'] and not train:
        plot_state_variables_dynamis(np.array(x_all))

    return y_arr, y_model


def call_load_u_y(constants):
    global u_train, y_train, u_test, y_test, load_u_y_bool
    (u_train, y_train) = _1_utils.load_u_y(constants)
    (u_test, y_test) = _1_utils.load_u_y(constants, train=False)
    load_u_y_bool = True


def paras_to_ABCD_swarm(params, constants):
    A_init = np.zeros((constants['state_num'], constants['state_num']))
    B_init = np.zeros((constants['state_num'], constants['input_num']))
    C_init = np.zeros((1, constants['state_num']))
    D_init = np.zeros((1, constants['input_num']))

    A, B, C, D = _1_utils.assgin_ABCD(A_init, B_init, C_init, D_init, params, case_nbr=constants['case_nbr'])

    sys = signal.StateSpace(A, B, C, D)
    sys_d = sys.to_discrete(constants['ts_sampling'])
    a = sys_d.A
    b = sys_d.B
    c = sys_d.C
    d = sys_d.D

    return a, b, c, d


def predict(pos, constants):
    y_train, y_train_pred = obj_func(pos, constants, train=True)
    ytest, y_test_pred = obj_func(pos, constants, train=False)
    return y_train, y_train_pred, ytest, y_test_pred


def load_pos():
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    all_pos_to_compare_abs = os.path.join(script_dir, 'inputs', 'pos_rscs.txt')
    with open(all_pos_to_compare_abs, 'r') as f:
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
