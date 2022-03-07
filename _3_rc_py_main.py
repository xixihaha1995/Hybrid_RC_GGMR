import sys, lmfit, _0_config, _2_optimization, _4_pyswarm
# Import PySwarms
import pyswarms as ps


if __name__ == "__main__":
    state_num = 7
    input_num = 12
    ts_sampling = 120
    _0_config.init(state_num, input_num, ts_sampling)
    _0_config.start = 0
    _0_config.end = 100

    if sys.argv[1] != 'swarm':
        para = _2_optimization.init_para()#rscs
        arg = _2_optimization.load_u_y()#u, y
        o1 = lmfit.minimize(_2_optimization.resid, para, args=arg, method='leastsq')
        lmfit.report_fit(o1)
        _2_optimization.plot(o1, arg[1])
    else:
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
        y_arr_pred = _4_pyswarm.predict(pos)
        _2_optimization.swarm_plot(_0_config.y_arr,y_arr_pred )



