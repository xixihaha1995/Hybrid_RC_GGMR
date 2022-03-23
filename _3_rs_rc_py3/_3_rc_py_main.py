import argparse, textwrap, time
import pyswarms as ps

import _1_utils, _2_pyswarm
from pyswarms.utils.plotters import plot_cost_history
from datetime import datetime

if __name__ == "__main__":
    all_log_dict = _1_utils.loadJSON("all_cases_log")
    this_log_dict ={}
    fmt = '%Y%m%d%H%M%S'
    this_log_dict['time'] = datetime.now().strftime(fmt)
    start_time = time.time()
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
    Use like: 
    --------------------------------
    this.py -arg [ts_sampling] [start] [end] [state_num] [input_num]...
    [dimensions] [n_particle] [iters]...
    [rc network, single = -1, cav = 0, room = 1, slab = 2, integrated = 3, slab_adj = 4, room_state = 5, room_sink_state = 6],[mode, optimization = 0, visual = 1]
    '''))
    parser.add_argument("-a", nargs='+', type=int, help="Specify args used for RC modeling")

    args = parser.parse_args()
    ts_sampling = args.a[0]
    start = args.a[1]
    end = args.a[2]
    state_num = args.a[3]
    input_num = args.a[4]
    dimensions = args.a[5]

    swarm_constants = {}
    swarm_constants['start'] = start
    swarm_constants['end'] = end
    swarm_constants['input_num'] = input_num
    swarm_constants['state_num'] = state_num
    swarm_constants['ts_sampling'] = ts_sampling
    swarm_constants['n_particles'] = args.a[6]
    swarm_constants['iters'] = args.a[7]
    swarm_constants['case_nbr'] = args.a[8]

    this_log_dict['arg'] = swarm_constants

    if args.a[9] == 0:
        # Hyperparameters for pyswarms
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # initialize rscs with energyplus manually calculated rscs, or matlab optimized rscs"
        rscs_init = _2_pyswarm.init_pos(args.a[8],swarm_constants['n_particles'])

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
        this_log_dict['elapsed_time'] = end_time - start_time
        if str(swarm_constants['case_nbr']) not in all_log_dict:
            all_log_dict[str(swarm_constants['case_nbr'])] = []

        all_log_dict[str(swarm_constants['case_nbr'])].append(this_log_dict)
        _1_utils.saveJSON(all_log_dict, "all_cases_log")
        plot_cost_history(cost_history)
        y_train, y_train_pred, ytest, y_test_pred = _2_pyswarm.predict(pos, swarm_constants)
        _1_utils.swarm_plot(y_train, y_train_pred, ytest, y_test_pred, swarm_constants)

    elif args.a[9] == 1:
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
