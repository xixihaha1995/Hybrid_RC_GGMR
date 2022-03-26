Directory structure:
```
Hybrid_GGMR_RC_Folder
│   quick_start.py
│   hybrid_ggmr_rc.py
└───RC
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
│   
└───GGMR_Folder
    │   file021.txt
    │   file022.txt
    │   ...
```

Usage: <br>
```python
from RC import _3_rc_py_main as RC_package
rc_obj = RC_package.RC_Class() # call this once

for time_idx in [66, 70]:
    print(rc_obj.predict(_time_idx = time_idx))
```
See more details in `quick_start.py`