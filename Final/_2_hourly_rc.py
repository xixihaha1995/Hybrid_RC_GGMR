import _0_utils as general_tool, numpy as np, pandas as pd

measure, hybrid, ggmr, rcm3, rcm2,rcm1 = general_tool.load_predicted_measure("all_models_save_5min.csv")
measure_hourly_all = general_tool.to_hourly(abs(measure))
hybrid_hourly_all = general_tool.to_hourly(abs(hybrid))
ggmr_hourly_all = general_tool.to_hourly(abs(ggmr))
rcm3_hourly_all = general_tool.to_hourly(abs(rcm3))
rcm2_hourly_all = general_tool.to_hourly(abs(rcm2))
rcm1_hourly_all = general_tool.to_hourly(abs(rcm1))

df = pd.DataFrame({"measure" : measure_hourly_all, "hybrid" : hybrid_hourly_all,
                   "ggmr":ggmr_hourly_all, "rc3":rcm3_hourly_all,
                   "rc2":rcm2_hourly_all,"rc1":rcm1_hourly_all})
df.to_csv(".csv", index=False)