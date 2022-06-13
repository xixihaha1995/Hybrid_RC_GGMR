import _0_utils as general_tool, numpy as np, pandas as pd

measure, hybrid,hybrid1, ggmr,ggmr2, ggmr1, rcm3, rcm2,rcm1 = general_tool.load_predicted_measure("all_models_save_5min.csv")
measure_hourly_all = general_tool.to_hourly(abs(measure))
hybrid_hourly_all = general_tool.to_hourly(abs(hybrid))
hybrid1_hourly_all = general_tool.to_hourly(abs(hybrid1))
ggmr_hourly_all = general_tool.to_hourly(abs(ggmr))
ggmr2_hourly_all = general_tool.to_hourly(abs(ggmr2))
ggmr1_hourly_all = general_tool.to_hourly(abs(ggmr1))
rcm3_hourly_all = general_tool.to_hourly(abs(rcm3))
rcm2_hourly_all = general_tool.to_hourly(abs(rcm2))
rcm1_hourly_all = general_tool.to_hourly(abs(rcm1))

# measure_hourly_all = general_tool.to_hourly(measure)
# hybrid_hourly_all = general_tool.to_hourly(hybrid)
# hybrid1_hourly_all = general_tool.to_hourly(hybrid1)
# ggmr_hourly_all = general_tool.to_hourly(ggmr)
# ggmr2_hourly_all = general_tool.to_hourly(ggmr2)
# ggmr1_hourly_all = general_tool.to_hourly(ggmr1)
# rcm3_hourly_all = general_tool.to_hourly(rcm3)
# rcm2_hourly_all = general_tool.to_hourly(rcm2)
# rcm1_hourly_all = general_tool.to_hourly(rcm1)

df = pd.DataFrame({"measure" : measure_hourly_all, "hybrid" : hybrid_hourly_all,"hybrid1" : hybrid1_hourly_all,
                   "ggmr":ggmr_hourly_all, "ggmr2":ggmr2_hourly_all, "ggmr1":ggmr1_hourly_all,
                   "rc3":rcm3_hourly_all,"rc2":rcm2_hourly_all,"rc1":rcm1_hourly_all})
df.to_csv("all_models_save_hourly_abs.csv", index=False)