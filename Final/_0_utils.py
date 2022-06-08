import numpy as np, os, pandas as pd, statistics, matplotlib.pyplot as plt
from scipy.stats import gmean

def load_all(filename):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    _measured_modeled_abs = os.path.join(script_dir, filename)
    _measured_modeled = pd.read_csv(_measured_modeled_abs)
    measured_modeled_arr = _measured_modeled.to_numpy()
    measure = measured_modeled_arr[:, 0]
    hybrid = measured_modeled_arr[:, 1]
    ggmr = measured_modeled_arr[:, 2]
    rc_model3 = measured_modeled_arr[:, 3]
    rc_model2 = measured_modeled_arr[:, 4]
    rc_model1 = measured_modeled_arr[:, 5]

    return measure,hybrid, ggmr, rc_model3, rc_model2,rc_model1

def load_predicted_measure(filename):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    _measured_modeled_abs = os.path.join(script_dir, filename)
    _measured_modeled = pd.read_csv(_measured_modeled_abs)
    measured_modeled_arr = _measured_modeled.to_numpy()
    measure = measured_modeled_arr[:, 0]
    hybrid = measured_modeled_arr[:, 1]
    ggmr = measured_modeled_arr[:, 2]
    rcm3 = measured_modeled_arr[:, 3]
    rcm2 = measured_modeled_arr[:, 4]
    rcm1 = measured_modeled_arr[:, 5]
    return measure, hybrid, ggmr, rcm3, rcm2,rcm1

def nrmse(measure, model):
    nom = (sum((measure - model) ** 2) / len(measure)) ** (1 / 2)
    if not isinstance(nom, float):
        nom = nom[0,0]
    mean = model.mean()
    denom = max(model) - min(model)
    # mean = abs(model).mean()
    # denom = (sum((model - mean) ** 2) / len(model)) ** (1 / 2)
    return nom *100 / denom

def cv_rmse(measure, model):
    new_model = []
    for num in model:
        if (np.isnan(num)):
            new_model.append(0)
        else:
            new_model.append(num)
    rmse = (sum((measure - new_model) ** 2) / len(measure)) ** (1 / 2)
    mean_measured = abs(measure).mean()
    return rmse * 100 / mean_measured


def mae(measure, model):
    new_model = []
    for num in model:
        if (np.isnan(num)):
            new_model.append(0)
        else:
            new_model.append(num)
    return sum(abs(measure - new_model)) / len(measure)

def mean_absolute_percentage_error(measure, model):
    nom = abs(measure - model) / abs(measure)
    # fig, ax = plt.subplots()
    # axTwn = ax.twinx()
    # ax.plot(measure, label = "Measured")
    # ax.plot(model, label = "Predicted")
    # axTwn.plot(nom, label = "error dist")
    # h1, l1 = ax.get_legend_handles_labels()
    # h2, l2 = axTwn.get_legend_handles_labels()
    # ax.legend(h1 + h2, l1 + l2)
    # plt.show()
    nom[nom == float('inf')] = 0
    new_nom = []
    for num in nom:
        if (np.isnan(num)):
            new_nom.append(0)
        else:
            new_nom.append(num)
    return sum(new_nom)*100 / len(measure)

def median_absolute_percentage_error(measure, model):
    nom = abs(measure - model) / abs(measure)
    return statistics.median(nom)*100

def geometric_median_absolute_percentage_error(measure, model):
    nom = abs(measure - model) / abs(measure)
    gmape = gmean(nom) * 100
    return gmape

def all_error(measure, model):
    nom = abs(measure - model) / abs(measure)
    return nom

def to_hourly(_5_min_model):
    ts_sampling = 5 * 60
    train_interval_steps = int(_5_min_model.shape[0] // (60*60 / ts_sampling))
    model = _5_min_model[:]
    model = model[:train_interval_steps * int(60 * 60 / ts_sampling) ]
    model = model.reshape((train_interval_steps, int(60 * 60 / ts_sampling)))
    hourly_predicted = np.sum(model, axis=1)

    return hourly_predicted