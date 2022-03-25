import time, os
import numpy as np
import pyswarms as ps
from . import _1_utils, _2_pyswarm
from pyswarms.utils import Reporter
from pyswarms.utils.plotters import plot_cost_history
from datetime import datetime

def predict(time_stamp_idx = 0):
    pass
#     load pos
#   output obj_function(pos, idx)

def train(_test_start = None, _end = None,  _n_particle = 2,_iters = 2,_case_nbr = 6, _ts = 300,_state_num = 6, _input_num = 9, _para_nums = 24 ):
    '''
    Use like:
    --------------------------------
    .train([start] [end] [rc network, single = -1, cav = 0, room = 1, slab = 2, integrated = 3, slab_adj = 4, room_state = 5, room_sink_state = 6]...
    [ts_sampling] [n_particle] [iters] [state_num] [input_num] [dimensions])
    '''
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


    all_log_dict = _1_utils.loadJSON("all_cases_log")
    this_log_dict ={}
    fmt = '%Y%m%d%H%M%S'
    this_log_dict['time'] = datetime.now().strftime(fmt)
    start_time = time.time()

    ts_sampling = _ts
    start = _test_start
    end = _end
    state_num = _state_num
    input_num = _input_num
    dimensions = _para_nums

    swarm_constants = {}
    swarm_constants['start'] = start
    swarm_constants['end'] = end
    swarm_constants['input_num'] = input_num
    swarm_constants['state_num'] = state_num
    swarm_constants['ts_sampling'] = ts_sampling
    swarm_constants['n_particles'] = _n_particle
    swarm_constants['iters'] = _iters
    swarm_constants['case_nbr'] =_case_nbr

    this_log_dict['arg'] = swarm_constants

    # Hyperparameters for pyswarms
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # initialize rscs
    rscs_init = _2_pyswarm.init_pos(_case_nbr,swarm_constants['n_particles'])

    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_constants['n_particles'], dimensions=dimensions,
                                        options=options,
                                        init_pos=rscs_init)
    # Perform optimization
    cost, pos = optimizer.optimize(_2_pyswarm.whole_swarm_loss, iters=swarm_constants['iters'],
                                   constants=swarm_constants)

    end_time = time.time()
    print(f'Elapsed time:{end_time - start_time}')
    cost_history = optimizer.cost_history

    this_log_dict['cost_hist'] = cost_history
    this_log_dict['pos'] = pos.tolist()
    _1_utils.append_list_as_row("optimized_pos.txt", pos.tolist())
    this_log_dict['elapsed_time'] = end_time - start_time
    if str(swarm_constants['case_nbr']) not in all_log_dict:
        all_log_dict[str(swarm_constants['case_nbr'])] = []
    all_log_dict[str(swarm_constants['case_nbr'])].append(this_log_dict)
    _1_utils.saveJSON(all_log_dict, "all_cases_log")
    plot_cost_history(cost_history)
    y_train, y_train_pred, ytest, y_test_pred = _2_pyswarm.predict(pos, swarm_constants)
    all_y_measured = np.concatenate((y_train, ytest)).reshape(-1,1)
    all_y_modeled = np.concatenate((y_train_pred, y_test_pred)).reshape(-1,1)
    all_measure_model = np.concatenate((all_y_measured, all_y_modeled), axis= 1)
    measure_predict_csv_abs = os.path.join(script_dir, 'outputs', 'measured_modeled.csv')
    np.savetxt(measure_predict_csv_abs, all_measure_model, delimiter=",", header= 'measured, modeled')
    _1_utils.swarm_plot(y_train, y_train_pred, ytest, y_test_pred, swarm_constants)

def visual_all(_test_start = None, _end = None,  _n_particle = 2,_iters = 2,_case_nbr = 6, _ts = 300,_state_num = 6, _input_num = 9, _para_nums = 24 ):
    swarm_constants = {}
    swarm_constants['start'] = _test_start
    swarm_constants['end'] = _end
    swarm_constants['input_num'] = _input_num
    swarm_constants['state_num'] = _state_num
    swarm_constants['ts_sampling'] = _ts
    swarm_constants['n_particles'] = _n_particle
    swarm_constants['iters'] = _iters
    swarm_constants['case_nbr'] =_case_nbr

    all_pos = _2_pyswarm.load_pos()
    all_pos_performance = []
    for i in range(len(all_pos)):
        cur_performance_dict = {}
        cur_performance_dict['title'] = all_pos[i]['title']
        cur_performance = []
        y_train, y_train_pred, ytest, y_test_pred = _2_pyswarm.predict(all_pos[i]['pos'], swarm_constants)
        cur_performance.append(y_train)
        cur_performance.append(y_train_pred)
        cur_performance.append(ytest)
        cur_performance.append(y_test_pred)
        cur_performance_dict['performance'] = cur_performance
        all_pos_performance.append(cur_performance_dict)
    _1_utils.pos_plot_all(all_pos_performance)
