import numpy as np
import _1_utils
import csv, os
from scipy import signal
import matplotlib.pyplot as plt

u_train = None
y_train = None
u_test = None
y_test = None
load_u_y_bool = False

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
