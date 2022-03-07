import sys, lmfit, _0_config, _2_optimization, _4_pyswarm
# Import PySwarms
import pyswarms as ps


if __name__ == "__main__":
    state_num = 7
    input_num = 12
    ts_sampling = 120
    _0_config.init(state_num, input_num, ts_sampling)
    _0_config.start = 0
    _0_config.end = 5040

    if sys.argv[1] == 'lmfit':
        para = _2_optimization.init_para()#rscs
        arg = _2_optimization.load_u_y()#u, y
        o1 = lmfit.minimize(_2_optimization.resid, para, args=arg, method='leastsq')
        lmfit.report_fit(o1)
        _2_optimization.plot(o1, arg[1])
    elif(sys.argv[1] == 'swarm'):
        _0_config.n_particles = int(sys.argv[2])
        _0_config.iters = int(sys.argv[3])
        rscs_init = _4_pyswarm.init_pos()
        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # Call instance of PSO
        dimensions = 23
        optimizer = ps.single.GlobalBestPSO(n_particles=_0_config.n_particles, dimensions=dimensions, options=options, init_pos= rscs_init)

        # Perform optimization
        cost, pos = optimizer.optimize(_4_pyswarm.whole_swarm_loss, iters= _0_config.iters)

        y_train_pred, y_test_pred = _4_pyswarm.predict(pos)
        _2_optimization.swarm_plot(_0_config.y_arr,y_train_pred, _0_config.y_arr_test,y_test_pred)

    elif(sys.argv[1] == 'swarm_visual'):
        all_pos = _4_pyswarm.load_pos()
        all_pos_performance =[]
        for i in range(len(all_pos)):
            cur_performance_dict = {}
            cur_performance_dict['title'] = all_pos[i]['title']
            cur_performance = []
            y_train, y_train_pred , ytest, y_test_pred  = _4_pyswarm.predict(all_pos[i]['pos'], measure= True)
            cur_performance.append(y_train)
            cur_performance.append(y_train_pred)
            cur_performance.append(ytest)
            cur_performance.append(y_test_pred)
            cur_performance_dict['performance'] = cur_performance
            all_pos_performance.append(cur_performance_dict)
        _2_optimization.pos_plot_all(all_pos_performance)






