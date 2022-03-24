Usage:
```
python  _3_rc_py_main.py -a [time step] [start time (th time step)] [end time (th time step)] [state_num] [input_num] [RsCs number] [particle_num] [iters_num][cases_num, single zone = -1, cav temp = 0, room temp = 1, slab temp = 2, integrated = 3, slab adj = 4, room as state = 5, room with sink as state = 6],[opt_mode, optimization = 0, visual = 1]
```
For example, to run the integrated case (room, sink as state variables) for Q_rad prediction, with time step = 300(seconds), start time = 0 th time step, end time = 2016th time step, state_num = 6, input_num = 9, paras_num =24, particle_num = 2, iters_num = 2, cases_num = 6, opt_mode = 0:

```python
python _3_rc_py_main.py -a 300 0 2016 6 9 24 2 2 6 0
```