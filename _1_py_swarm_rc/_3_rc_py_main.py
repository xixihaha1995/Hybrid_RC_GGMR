import sys, _1_utils, _2_pyswarm
import pyswarms as ps

if __name__ == "__main__":
    nl = '\n'
    print(f'usage: _4_rc_py_main.py  [argvs]{nl} example1: _3_rc_py_main.py swarm number_of_particles number_of_iters{nl} example2: _3_rc_py_main.py swarm_visual')
    state_num = 7
    input_num = 12
    ts_sampling = 120

    start = 5040 * 0
    end = 5040 * 1
    dimensions = 23

    # _mp_state_num = Value('d', 3.14)
    swarm_constants = {}
    swarm_constants['start'] = start
    swarm_constants['end'] = end
    swarm_constants['input_num'] = input_num
    swarm_constants['state_num'] = state_num
    swarm_constants['ts_sampling'] = ts_sampling

    if (sys.argv[1] == 'swarm'):
        swarm_constants['n_particles'] = int(sys.argv[2])
        # Hyperparameters for pyswarms
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # initialize rscs with energyplus manually calculated rscs, or matlab optimized rscs"
        rscs_init = _2_pyswarm.init_pos(swarm_constants['n_particles'])
        swarm_constants['iters'] = int(sys.argv[3])
        optimizer = ps.single.GlobalBestPSO(n_particles=swarm_constants['n_particles'], dimensions=dimensions, options=options,
                                            init_pos=rscs_init)
        # Perform optimization
        cost, pos = optimizer.optimize(_2_pyswarm.whole_swarm_loss, iters=swarm_constants['iters'],n_processes=4, constants = swarm_constants)

        y_train, y_train_pred, ytest, y_test_pred  = _2_pyswarm.predict(pos, swarm_constants)
        _1_utils.swarm_plot(y_train, y_train_pred, ytest, y_test_pred, swarm_constants)

    elif (sys.argv[1] == 'swarm_visual'):
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
