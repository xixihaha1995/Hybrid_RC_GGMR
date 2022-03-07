def init(_stat_no, _input_no, _ts):
    global state_num, input_num, ts_sampling
    state_num = _stat_no
    input_num = _input_no
    ts_sampling = _ts
    u_arr = None
    y_arr = None
    n_particles = None