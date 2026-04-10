# data/dataset.py
import os, glob
import numpy as np
from PIL import Image

from .resample import resample_centerline_csv
from .nrrd_utils import read_and_resample_nrrd


# ----------------------------- helpers -----------------------------
def _pad_stack(chs):
    """Pad 2D feature maps to the max HxW in batch and stack to (C,H,W)."""
    H = max(a.shape[0] for a in chs)
    W = max(a.shape[1] for a in chs)
    out = []
    for a in chs:
        buf = np.zeros((H, W), dtype=np.float32)
        buf[:a.shape[0], :a.shape[1]] = a.astype(np.float32)
        out.append(buf)
    return np.ascontiguousarray(np.stack(out, axis=0), dtype=np.float32)


def _load_npz(root, pid, channels):
    """Load engineered 2D features from two npz (front/side)."""
    fF = os.path.join(root, pid, "front_features.npz")
    fS = os.path.join(root, pid, "side_features.npz")
    if not (os.path.exists(fF) and os.path.exists(fS)):
        return None
    F = np.load(fF)
    S = np.load(fS)

    front = _pad_stack([F[ch].astype(np.float32) for ch in channels])
    side = _pad_stack([S[ch].astype(np.float32) for ch in channels])

    def norm(x):
        x = x.copy()
        for i in range(x.shape[0]):
            v = x[i]
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            if hi <= lo:
                hi, lo = float(v.max()), float(v.min())
            v = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
            x[i] = v.astype(np.float32)
        return x

    out = np.concatenate([norm(front), norm(side)], axis=0)  # (2C,H,W)
    return np.ascontiguousarray(out, dtype=np.float32)


def _load_png(root, pid):
    """Fallback: load grayscale surface projections from two PNGs."""
    a = os.path.join(root, pid, "surface_front.png")
    b = os.path.join(root, pid, "surface_side.png")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    f = np.array(Image.open(a).convert("L"), dtype=np.float32) / 255.0
    s = np.array(Image.open(b).convert("L"), dtype=np.float32) / 255.0
    H = max(f.shape[0], s.shape[0])
    W = max(f.shape[1], s.shape[1])
    F = np.zeros((H, W), dtype=np.float32)
    S = np.zeros((H, W), dtype=np.float32)
    F[:f.shape[0], :f.shape[1]] = f
    S[:s.shape[0], :s.shape[1]] = s
    return np.ascontiguousarray(np.stack([F, S], axis=0), dtype=np.float32)


def _first_existing(paths):
    """Return first existing path, else None."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _candidate_patient_roots(data_root, pid):
    """
    Support both:
      <data_root>/<PID>/<PID>/...
      <data_root>/<PID>/...
    """
    a = os.path.join(data_root, pid, pid)   # two-level
    b = os.path.join(data_root, pid)        # one-level
    return [a, b]


def _select_ct_file(ct_dir, prefer_filename=None):
    """
    In ct_dir, if prefer_filename exists -> return it;
    else search *.nrrd and return the largest file.
    """
    if prefer_filename:
        prefer_path = os.path.join(ct_dir, prefer_filename)
        if os.path.exists(prefer_path):
            return prefer_path

    cand = sorted(glob.glob(os.path.join(ct_dir, "*.nrrd")))
    if not cand:
        raise FileNotFoundError(f"No .nrrd found in: {ct_dir}")

    # choose largest file (most likely original volume)
    cand.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return cand[0]


# ----------------------- main dataset loader ------------------------
class VesselSeqSampleLoader:
    """
    Robust sample loader that:
      - auto-discovers CT .nrrd under Origin data/ (filename-agnostic)
      - supports one/two-level patient folder layouts
      - loads 2D projection features (.npz or fallback .png)
      - packs targets to vector: (S*N*4 + S)
    """
    def __init__(self, cfg, demo_map):
        self.cfg = cfg
        self.demo_map = demo_map
        self.S = int(cfg["num_segments"])
        self.N = int(cfg["seq_points"])
        self.segment_names = cfg["segment_names"]
        self.channels = cfg.get("projection_channels", ["bin", "thickness_mm", "depth_mm", "edt_mm"])

        # names from cfg
        self.origin_subdir = cfg.get("origin_subdir", "Origin data")
        self.tables_subdir = cfg.get("tables_subdir", "Tables")
        self.ct_prefer_name = cfg.get("ct_filename", None)  # optional hint, not required

        # data roots
        self.data_root = cfg["data_root"]
        self.proj_root = cfg.get("projections_root", None)

    # --------------------- path resolvers ---------------------
    def _find_origin_dir(self, pid):
        roots = _candidate_patient_roots(self.data_root, pid)
        cand = [os.path.join(r, self.origin_subdir) for r in roots]
        d = _first_existing(cand)
        if d is None:
            raise FileNotFoundError(f"Origin data folder not found for PID={pid}. Tried: {cand}")
        return d

    def _find_tables_dir(self, pid):
        roots = _candidate_patient_roots(self.data_root, pid)
        cand = [os.path.join(r, self.tables_subdir) for r in roots]
        d = _first_existing(cand)
        if d is None:
            raise FileNotFoundError(f"Tables folder not found for PID={pid}. Tried: {cand}")
        return d

    # --------------------- actual loaders ---------------------
    def load_ct(self, pid):
        """
        Read any .nrrd CT file under .../Origin data/, resample to (2mm)^3,
        clip to [-200, 800] HU and normalize to [0,1], output shape (1,D,H,W).
        """
        ct_dir = self._find_origin_dir(pid)
        ct_path = _select_ct_file(ct_dir, prefer_filename=self.ct_prefer_name)

        arr, _ = read_and_resample_nrrd(ct_path, out_spacing=(2.0, 2.0, 2.0), is_label=False)

        # HU clamp & normalize
        vol = np.clip(arr.astype(np.float32), -200.0, 800.0)
        vol = (vol + 200.0) / 1000.0
        vol = np.expand_dims(vol, 0).astype(np.float32)  # (1,D,H,W)
        return np.ascontiguousarray(vol, dtype=np.float32)

    def load_imgs(self, pid):
        """
        Load 2D projections/features:
          1) <projections_root>/<PID>/{front_features.npz, side_features.npz}
          2) fallback: <projections_root>/<PID>/{surface_front.png, surface_side.png}
        """
        if self.proj_root is None:
            raise FileNotFoundError("projections_root is not set in config.")

        x = _load_npz(self.proj_root, pid, self.channels)
        if x is None:
            x = _load_png(self.proj_root, pid)
            if x is None:
                raise FileNotFoundError(f"Missing 2D features for {pid} under {os.path.join(self.proj_root, pid)}")
        return x  # (C,H,W), C = 2 * len(channels) or 2

    def load_targets(self, pid):
        """
        Pack targets to vector: concat over segments:
          coords_diam: (S,N,4) -> flatten to S*N*4
          length:      (S,)    -> append to end
        """
        base = self._find_tables_dir(pid)

        seqs, lens = [], []
        for fn in self.segment_names:
            csv_path = os.path.join(base, fn)
            seq, L = resample_centerline_csv(csv_path, n_points=self.N)  # seq: (N,4) = (x,y,z,diam)
            seqs.append(seq)
            lens.append(L)

        arr = np.stack(seqs, axis=0).astype(np.float32)  # (S,N,4)
        L = np.asarray(lens, dtype=np.float32)           # (S,)
        vec = np.concatenate([arr.reshape(-1), L], axis=0).astype(np.float32)
        return np.ascontiguousarray(vec, dtype=np.float32)

    def __call__(self, pid):
        demo = self.demo_map.get(pid, None)
        if demo is None:
            raise KeyError(f"Demographics missing for PID={pid}")

        return {
            "pid": pid,
            "ct": self.load_ct(pid),           # (1,D,H,W) float32
            "img2d": self.load_imgs(pid),     # (C,H,W)   float32
            "demo": demo.astype(np.float32),  # (K,)      float32
            "target": self.load_targets(pid)  # (S*N*4+S,)
        }
