from evaluate_dice import rq1_flip_rates
p = "derby@0"
mt =['RandomForest']
pro = [p]
me = ['best']
x = rq1_flip_rates(mt,pro,me)
print(x)

import pandas as pd

df = pd.read_csv(f"/Users/joony/Desktop/EMSE/eval-actionable-guidance/experiments/{p}/RandomForest/random/DiCE_all_100_nofeat.csv")
unique_count = df["test_idx"].nunique()   # NaNs ignored by default
print(unique_count)
