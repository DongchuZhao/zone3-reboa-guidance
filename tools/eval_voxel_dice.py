import os, glob, csv, numpy as np, SimpleITK as sitk
from skimage.morphology import ball
from scipy.ndimage import binary_dilation

def load_gt_mask(nrrd_path):
    img = sitk.ReadImage(nrrd_path)      # e.g., .../Vessel.seg.nrrd
    arr = sitk.GetArrayFromImage(img).astype(bool)  # [Z,Y,X]
    spacing = img.GetSpacing()[::-1]     # (Z,Y,X) mm
    origin  = img.GetOrigin()[::-1]
    direction = img.GetDirection()
    return arr, spacing, origin, direction

def voxelize_tube(points_mm, diams_mm, shape, spacing):
    # points_mm: [N,3] in (X,Y,Z) mm; spacing: (Z,Y,X) mm
    vol = np.zeros(shape, dtype=bool)
    Z, Y, X = shape
    sz, sy, sx = spacing
    for (xmm, ymm, zmm), dmm in zip(points_mm, diams_mm):
        # [internal note]
        zz = int(round(zmm / sz))
        yy = int(round(ymm / sy))
        xx = int(round(xmm / sx))
        if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
            # [internal note]
            rz, ry, rx = (dmm/2)/np.array([sz, sy, sx])
            r = int(np.ceil(max(rx, ry, rz)))
            se = ball(max(r,1))
            # [internal note]
            seed = np.zeros_like(vol); seed[zz,yy,xx] = True
            vol |= binary_dilation(seed, se)
    return vol

def dice(a, b):
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return (2.0*inter/denom) if denom>0 else 1.0

if __name__ == "__main__":
    # [internal note]
    PID = "SXXXX"
    gt_nrrd   = f"data/{PID}/Vessel.seg.nrrd"
    pred_pts  = f"outputs/predict/{PID}/pred_centerline.npy"  # [N,3] (X,Y,Z) in mm
    pred_diam = f"outputs/predict/{PID}/pred_diameter.npy"    # [N] in mm

    gt, spacing, *_ = load_gt_mask(gt_nrrd)
    pts = np.load(pred_pts)
    dmm = np.load(pred_diam)
    pred_vol = voxelize_tube(pts, dmm, gt.shape, spacing)
    d = dice(pred_vol, gt)
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/dice_voxel.csv","a",newline="") as f:
        w=csv.writer(f); w.writerow([PID, float(d)])
    print(f"{PID} Dice(voxel) = {d:.4f}  -> outputs/metrics/dice_voxel.csv")
