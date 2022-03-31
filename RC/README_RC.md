Directory structure:
```
Hybrid_GGMR_RC_Folder
│   README_RC.md
│   report.log #this is auto-generated from pyswarms
│   quick_start.py
│   hybrid_ggmr_rc.py
|
└───RC_prediction
│   │   abcd.json
│   │   call_rc_function.py
└───RC_training
│   │   _1_utils.py
│   │   _2_pyswarm.py
│   │   _3_rc_py_main.py
│   └───inputs
│       │   file111.txt
│       │   file112.txt
│       │   ...
│       outputs
│       │   file111.txt
│       │   file112.txt
│       │   ...
|
└───GGMR_Folder
    │   file021.txt
    │   file022.txt
    │   ...
```

Usage: <br>
```python
from multiprocessing import freeze_support
from RC_training import _2_pyswarm as RC_utils
from RC_prediction import call_rc_function
if __name__ == '__main__':
    freeze_support()
    
    ...
    #The above codes is to show the expected input array for RC prediction
    target_time_idx = 782
    warming_up_steps = 15
    u_measured = RC_utils.warming_input_demo(time_idx = target_time_idx, seg_length= warming_up_steps)
    #The above codes is to show the expected input array for RC prediction

    print(f'u_measured shape:{u_measured.shape}')
    # print(f'u_measured:{u_measured}')
    rc_function = call_rc_function.RC_Prediction()
    print(f'RC prediction with pre-calcualted abcd:{rc_function.predict(u_measured)} Watts load')
```
Please see more details in `quick_start.py`.