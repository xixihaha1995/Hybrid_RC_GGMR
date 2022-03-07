import sys, lmfit, _1_config, _2_utils, _3_pyswarm
# Import PySwarms
import pyswarms as ps

if __name__ == "__main__":
    nl = '\n'
    print(f'usage: _3_rc_py_main.py  [argvs]{nl} example1: _3_rc_py_main.py swarm number_of_particles number_of_iters{nl} example2: _3_rc_py_main.py swarm_visual')
    state_num = 7
    input_num = 12
    ts_sampling = 120
    _1_config.init(state_num, input_num, ts_sampling)
    _1_config.start = 0
    _1_config.end = 5040

    if (sys.argv[1] == 'swarm'):
        _1_config.n_particles = int(sys.argv[2])
        # number of rscs = 23
        dimensions = 23
        # Hyperparameters for pyswarms
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # initialize rscs with energyplus manually calculated rscs, or matlab optimized rscs"
        rscs_init = _3_pyswarm.init_pos()
        _1_config.iters = int(sys.argv[3])
        optimizer = ps.single.GlobalBestPSO(n_particles=_1_config.n_particles, dimensions=dimensions, options=options,
                                            init_pos=rscs_init)
        # Perform optimization
        cost, pos = optimizer.optimize(_3_pyswarm.whole_swarm_loss, iters=_1_config.iters)
        y_train, y_train_pred, ytest, y_test_pred  = _3_pyswarm.predict(pos)
        _2_utils.swarm_plot(y_train, y_train_pred, ytest, y_test_pred)

    elif (sys.argv[1] == 'swarm_visual'):
        all_pos = _3_pyswarm.load_pos()
        all_pos_performance = []
        for i in range(len(all_pos)):
            cur_performance_dict = {}
            cur_performance_dict['title'] = all_pos[i]['title']
            cur_performance = []
            y_train, y_train_pred, ytest, y_test_pred = _3_pyswarm.predict(all_pos[i]['pos'], measure=True)
            cur_performance.append(y_train)
            cur_performance.append(y_train_pred)
            cur_performance.append(ytest)
            cur_performance.append(y_test_pred)
            cur_performance_dict['performance'] = cur_performance
            all_pos_performance.append(cur_performance_dict)
        _2_utils.pos_plot_all(all_pos_performance)
