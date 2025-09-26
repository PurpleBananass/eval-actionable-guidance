import pandas as pd

# Read CSV
df = pd.read_csv("/home/joony/saner/jit2/eval-actionable-guidance/evaluations/feasibility/mahalanobis/RF_DiCE_random_best.csv")


# Filter out values greater than 5 (treat them as outliers)
df_filtered = df.where(df <= 5)

# Compute column-wise averages without outliers
col_avgs = df_filtered.mean()

print("Column-wise averages (outliers >5 removed):")
print(col_avgs)