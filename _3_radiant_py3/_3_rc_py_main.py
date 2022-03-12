import argparse, textwrap

import pyswarms as ps
import sys, _1_utils, _2_pyswarm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
    Use like: 
    --------------------------------
    this.py -arg [ts_sampling] [start] [end] [state_num] [input_num] [dimensions] [n_particle] [iters] [rc network, cav = 0, room = 1, slab = 2, integrated = 3]
    or
    this.py -v 
    '''))
    parser.add_argument("-a", nargs='+', type=int, help="Specify args used for rc network")
    parser.add_argument("-v", '--visual', action="store_true", help="Visualize all pos performance")
    args = parser.parse_args()
    ts_sampling = args.a[0]
    start = args.a[1]
    end = args.a[2]
    state_num = args.a[3]
    input_num = args.a[4]
    dimensions = args.a[5]

    # _mp_state_num = Value('d', 3.14)
    swarm_constants = {}
    swarm_constants['start'] = start
    swarm_constants['end'] = end
    swarm_constants['input_num'] = input_num
    swarm_constants['state_num'] = state_num
    swarm_constants['ts_sampling'] = ts_sampling
    swarm_constants['n_particles'] = args.a[6]
    swarm_constants['iters'] = args.a[7]
    swarm_constants['case_nbr'] = args.a[8]

    if not args.visual:
        # Hyperparameters for pyswarms
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # initialize rscs with energyplus manually calculated rscs, or matlab optimized rscs"
        rscs_init = _2_pyswarm.init_pos(args.a[8],swarm_constants['n_particles'])

        optimizer = ps.single.GlobalBestPSO(n_particles=swarm_constants['n_particles'], dimensions=dimensions,
                                            options=options,
                                            init_pos=rscs_init)
        # Perform optimization
        cost, pos = optimizer.optimize(_2_pyswarm.whole_swarm_loss, iters=swarm_constants['iters'], n_processes=3,
                                       constants=swarm_constants)

        y_train, y_train_pred, ytest, y_test_pred = _2_pyswarm.predict(pos, swarm_constants)
        _1_utils.swarm_plot(y_train, y_train_pred, ytest, y_test_pred, swarm_constants)

    elif args.visual:
        all_pos = _2_pyswarm.load_pos()
        all_pos_performance = []
        for i in range(len(all_pos)):
            cur_performance_dict = {}
            cur_performance_dict['title'] = all_pos[i]['title']
            cur_performance = []
            y_train, y_train_pred, ytest, y_test_pred = _2_pyswarm.predict(all_pos[i]['pos'], measure=True)
            cur_performance.append(y_train)
            cur_performance.append(y_train_pred)
            cur_performance.append(ytest)
            cur_performance.append(y_test_pred)
            cur_performance_dict['performance'] = cur_performance
            all_pos_performance.append(cur_performance_dict)
        _1_utils.pos_plot_all(all_pos_performance)
