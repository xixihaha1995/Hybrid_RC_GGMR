def init(_stat_no, _input_no, _ts):
    global state_num, input_num, ts_sampling, start, end, u_arr, y_arr, n_particles, iters,u_arr_test, y_arr_measure
    state_num = _stat_no
    input_num = _input_no
    ts_sampling = _ts
    start = None
    end = None
    u_arr = None
    y_arr = None
    n_particles = None
    iters = None
    u_arr_test = None
    y_arr_test = None