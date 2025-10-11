from evaluate_dice import rq1_flip_rates
p = "activemq@0"
mt =['RandomForest']
pro = [p]
me = ['random']
x = rq1_flip_rates(mt,pro,me)
print(x)

import pandas as pd

df = pd.read_csv(f"/Users/joony/Desktop/new/eval-actionable-guidance/experiments/activemq@0/RandomForest/kfeature/DiCE_all_1_max5feat.csv")
unique_count = df["test_idx"].nunique()   # NaNs ignored by default
print(unique_count)
