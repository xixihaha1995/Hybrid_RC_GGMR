import os, pandas as pd, json

def load_rc_y_y():
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    _measured_modeled_abs = os.path.join(script_dir, 'data', '6_measured_modeled.csv')
    _measured_modeled = pd.read_csv(_measured_modeled_abs)
    measured_modeled_arr = _measured_modeled.to_numpy()
    rc_y = measured_modeled_arr[:, 1]
    y = measured_modeled_arr[:, 0]
    return rc_y, y

def loadJSONFromOutputs(name):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'outputs',name + '.json'), 'r') as f:
        testDict = json.loads(f.read())
    return testDict

def cvrmse_cal(measure, predict, mean_measured):
    rmse = (sum((measure - predict) ** 2) / len(measure)) ** (1 / 2)
    cvrmse = rmse * 100 / mean_measured
    return cvrmse

def saveJSON(data, name):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    with open(os.path.join(script_dir, 'outputs',name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)