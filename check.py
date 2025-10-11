from pathlib import Path
import pandas as pd
from hyparams import EXPERIMENTS  # same place your eval imports it

projects = ["activemq@0","camel@0"]          # put a couple you ran
models   = ["RandomForest","SVM"]            # match folder names exactly
methods  = ["kfeatures","random"]            # whatever you used when generating
total_cfs = 100                              # must match generator arg
max_features = 5                             # must match generator arg

for p in projects:
    for m in models:
        for meth in methods:
            path = Path(EXPERIMENTS)/p/meth and ""  # just to see structure
            path = Path(EXPERIMENTS) / p / m / meth / f"DiCE_all_{total_cfs}_max{max_features}feat.csv"
            exists = path.exists()
            print(f"{path} -> exists={exists}")
            if exists:
                print("  size:", path.stat().st_size, "bytes")
                try:
                    df = pd.read_csv(path, nrows=3)
                    print("  columns:", list(df.columns)[:12])
                    print(df.head(2))
                except Exception as e:
                    print("  failed to read:", e)
