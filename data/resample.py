import numpy as np, pandas as pd
from scipy.interpolate import interp1d

DEFAULT_COLS = {"r":"RAS_R","a":"RAS_A","s":"RAS_S","dist":"Distance","diam":"Diameter (CE)"}

def _pick_diam_col(df):
    if DEFAULT_COLS["diam"] in df.columns: return DEFAULT_COLS["diam"]
    for c in ["Diameter","Diameter_mm","diameter","diam"]:
        if c in df.columns: return c
    raise KeyError("Diameter column not found; expected 'Diameter (CE)' or similar.")

def resample_centerline_csv(csv_path, n_points=64, cols_map=None):
    df = pd.read_csv(csv_path)
    if cols_map is None:
        cols_map = DEFAULT_COLS.copy(); cols_map["diam"] = _pick_diam_col(df)
    r = df[cols_map["r"]].to_numpy(np.float32)
    a = df[cols_map["a"]].to_numpy(np.float32)
    s = df[cols_map["s"]].to_numpy(np.float32)
    dist = df[cols_map["dist"]].to_numpy(np.float32) if cols_map["dist"] in df.columns else None
    if dist is None: dist = np.arange(len(r), dtype=np.float32)
    diam = df[cols_map["diam"]].to_numpy(np.float32)

    order = np.argsort(dist)
    r,a,s,dist,diam = r[order],a[order],s[order],dist[order],diam[order]

    if len(dist) < 2 or float(dist[-1]) <= 0:
        out = np.zeros((n_points,4), np.float32)
        if len(r):
            out[:,0]=r[0]; out[:,1]=a[0]; out[:,2]=s[0]; out[:,3]=diam.mean() if len(diam) else 0.0
        return out, 0.0

    target = np.linspace(0.0, float(dist[-1]), num=n_points, dtype=np.float32)
    fr,fa,fs,fd = (interp1d(dist, x, kind="linear", fill_value="extrapolate") for x in (r,a,s,diam))
    out = np.stack([fr(target), fa(target), fs(target), fd(target)], axis=1).astype(np.float32)
    return out, float(dist[-1])
