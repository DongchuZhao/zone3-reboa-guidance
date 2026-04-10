import os, yaml, numpy as np, SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm
from pathlib import Path

def _mm_per_index(spacing_xyz, axis_idx):
    # SimpleITK spacing = (sx, sy, sz); array = (z, y, x)
    if axis_idx == 0: return spacing_xyz[2]  # along z -> sz
    if axis_idx == 1: return spacing_xyz[1]  # along y -> sy
    return spacing_xyz[0]                    # along x -> sx

def _normalize_to_uint8(x):
    x = np.asarray(x, dtype=np.float32)
    if np.all(x == 0): return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo: hi = x.max(); lo = x.min()
    x = (np.clip(x, lo, hi) - lo) / max(hi - lo, 1e-6)
    return (x * 255).astype(np.uint8)

def _first_hit_depth_mm(mask, axis, mm_per_idx):
    arr = np.moveaxis(mask, axis, 0)  # (A, *others)
    nz = arr > 0
    anycol = nz.any(axis=0)
    csum = np.cumsum(nz, axis=0)
    idx = np.argmax(csum > 0, axis=0)
    idx = np.where(anycol, idx, 0)
    return idx.astype(np.float32) * mm_per_idx

def build_features(arr, spacing_xyz, view):
    mask = (arr > 0).astype(np.uint8)
    if view == "front":
        axis = 1; sampling_2d = (spacing_xyz[2], spacing_xyz[0])  # (sz, sx)
    else:
        axis = 2; sampling_2d = (spacing_xyz[2], spacing_xyz[1])  # (sz, sy)
    mm = _mm_per_index(spacing_xyz, axis)

    bin2d = mask.max(axis=axis).astype(np.uint8)
    thickness_mm = mask.sum(axis=axis).astype(np.float32) * mm
    depth_mm = _first_hit_depth_mm(mask, axis=axis, mm_per_idx=mm)
    edt_mm = ndi.distance_transform_edt(bin2d, sampling=sampling_2d).astype(np.float32)

    return {"bin":bin2d, "thickness_mm":thickness_mm, "depth_mm":depth_mm, "edt_mm":edt_mm, "sampling_2d":sampling_2d}

def save_npz(out_dir, view, feats):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{view}_features.npz"),
                        bin=feats["bin"], thickness_mm=feats["thickness_mm"],
                        depth_mm=feats["depth_mm"], edt_mm=feats["edt_mm"],
                        sampling_2d=np.array(feats["sampling_2d"], dtype=np.float32))

def save_previews(out_dir, view, feats):
    for k in ["bin","thickness_mm","depth_mm","edt_mm"]:
        arr = feats[k]
        img = (arr*255).astype(np.uint8) if k == "bin" else _normalize_to_uint8(arr)
        Image.fromarray(img).save(os.path.join(out_dir, f"{view}_{k}.png"))

def process_pid(pid, cfg):
    seg_path = os.path.join(cfg["data_root"], pid, pid, cfg["models_subdir"], cfg["surface_seg_filename"])
    if not os.path.exists(seg_path):
        print(f"[WARN] Surface.seg missing: {seg_path}")
        return False
    img = sitk.ReadImage(seg_path)
    spacing = img.GetSpacing()
    arr = sitk.GetArrayFromImage(img)
    out_dir = os.path.join(cfg["projections_root"], pid)
    F = build_features(arr, spacing, "front")
    S = build_features(arr, spacing, "side")
    save_npz(out_dir, "front", F)
    save_npz(out_dir, "side", S)
    save_previews(out_dir, "front", F)
    save_previews(out_dir, "side", S)
    return True

if __name__ == "__main__":
    CFG = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
    ids = [d for d in os.listdir(cfg["data_root"]) if d.startswith(("S","SCT","ASCT"))]
    ok = 0
    for pid in tqdm(ids, desc="Projections (rich)"):
        if process_pid(pid, cfg): ok += 1
    print(f"Generated projections for {ok}/{len(ids)} -> {cfg['projections_root']}")
