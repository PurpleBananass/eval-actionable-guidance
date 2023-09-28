# %% 
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd

sk = importr('ScottKnottESD')
data = pd.DataFrame(
    {
        "TechniqueA": [5, 1, 4],
        "TechniqueB": [6, 8, 3],
        "TechniqueC": [7, 10, 15],
        "TechniqueD": [7, 10.1, 15],
    }
)

# %%
r_sk = sk.sk_esd(data)

# Convert 'IntVector' to Python List and subtract 1 from each element
column_order = [x - 1 for x in list(r_sk[3])]

# Convert 'IntVector' to Python List for ranks
rank_list = [int(x) for x in list(r_sk[1])]

# Create DataFrames
ranking = pd.DataFrame(
    {
        "technique": [data.columns[i] for i in column_order],
        "rank": rank_list,
    }
) # long format

ranking_wide = pd.DataFrame(
    [rank_list], columns=[data.columns[i] for i in column_order]
) # wide format

# %%
ranking_wide
# %%
