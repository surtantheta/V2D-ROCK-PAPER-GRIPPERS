#!/usr/bin/env python3
"""sha256 of the packed parquet's row_id + x/y/z (commit column excluded), as validator 05 defined it."""
import hashlib, sys
import pandas as pd
d = pd.read_parquet(sys.argv[1])
h = hashlib.sha256()
h.update("\n".join(d["row_id"].astype(str)).encode())
h.update(d[["x", "y", "z"]].to_numpy("float64").tobytes())
print(h.hexdigest())
