# Output/*/LIMEHPO_ 에 있는 모든 csv 파일을 읽어서, 각각 DataFrame으로 만들고, importance_ratio를 DataFrame마다 더한 값을 모든 파일에 대해 구한다.

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
output_dir = Path('Output')
for d in output_dir.iterdir():
    if d.is_dir():
        if (d / 'LIMEHPO_').exists():
            
            df = pd.DataFrame()
            for f in (d / 'LIMEHPO_').iterdir():
                if f.suffix == '.csv':
                    df_csv = pd.read_csv(f)
                    sum_ratio = df_csv['importance_ratio'].sum()
                    df = pd.concat([df, pd.DataFrame({'sum_ratio': sum_ratio}, index=[f.name])])
            print(f"{d.name} | {df['sum_ratio'].mean():.2f}")
            
# %%
