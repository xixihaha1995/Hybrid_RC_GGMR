import lmfit, _0_config, _2_optimization


if __name__ == "__main__":
    state_num = 7
    input_num = 12
    ts_sampling = 120
    _0_config.init(state_num, input_num, ts_sampling)

    para = _2_optimization.init_para()#rscs
    arg = _2_optimization.load_u_y()#u, y

    o1 = lmfit.minimize(_2_optimization.resid, para, args=arg, method='leastsq')
    # # print("# Fit using leastsq:")
    _2_optimization.plot(o1, arg[1])
    lmfit.report_fit(o1)

