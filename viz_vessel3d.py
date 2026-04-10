# -*- coding: utf-8 -*-
"""
viz_vessel3d.py
[internal]
  - [internal]
  - [internal]
  - [internal]
    Chamfer Distance、Tube Dice（[internal]
[internal]
  - outdir/figure.html    ([internal]
  - outdir/metrics.json   ([internal]
  - outdir/metrics.csv    ([internal]
[internal]
  python viz_vessel3d.py --pred path/to/pred.csv --gt path/to/gt.csv --outdir outputs/viz/S99240

[internal]
  1) CSV：[internal]
  2) NPZ：keys = {'points': (N,3), 'diameters'[see source]'seg_id'

[internal]
  pip install numpy pandas scipy scikit-image plotly

[internal]
  - [internal]
  - 3D[internal]
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion, distance_transform_edt
import plotly.graph_objects as go

# [internal note]

def _infer_cols(df):
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        raise KeyError(f"[see source]")
    x = pick('x')
    y = pick('y')
    z = pick('z')
    d = pick('diameter','d','diam_mm','radius_mm')  # [internal]
    seg = cols.get('seg_id', None)
    return x,y,z,d,seg

def load_path(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        x,y,z,d,seg = _infer_cols(df)
        arr = df[[x,y,z]].to_numpy(dtype=float)
        diam = df[d].to_numpy(dtype=float)
        if 'radius' in d.lower():
            diam = diam * 2.0
        seg_id = df[seg].to_numpy() if seg else None
        return arr, diam, seg_id
    elif path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        points = data['points'].astype(float)
        diam = data['diameters'].astype(float)
        seg_id = data['seg_id'] if 'seg_id' in data.files else None
        return points, diam, seg_id
    else:
        raise ValueError(f"[see source]")

def split_by_seg(points, diam, seg_id):
    if seg_id is None:
        return [(points, diam, 0)]
    out = []
    for sid in np.unique(seg_id):
        mask = seg_id == sid
        out.append((points[mask], diam[mask], int(sid)))
    return out

def arclength(points):
    if len(points)<2: return np.array([0.0]*len(points))
    seg = np.linalg.norm(np.diff(points,axis=0),axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    return s

def resample_curve(points, values, n=200):
    """[see source]"""
    s = arclength(points)
    if s[-1] < 1e-6:
        # [internal note]
        return points.copy(), values.copy()
    ss = np.linspace(0, s[-1], n)
    # [internal note]
    xyz = []
    for i in range(3):
        f = interp1d(s, points[:,i], kind='linear', bounds_error=False, fill_value=(points[0,i], points[-1,i]))
        xyz.append(f(ss))
    xyz = np.stack(xyz, axis=1)
    # [internal note]
    fv = interp1d(s, values, kind='linear', bounds_error=False, fill_value=(values[0], values[-1]))
    vv = fv(ss)
    return xyz, vv

# [internal note]

def _frame_from_tangent(t):
    """[see source]"""
    # [internal note]
    ref = np.array([0,0,1.0])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0,1.0,0])
    n = np.cross(t, ref)
    n /= (np.linalg.norm(n) + 1e-12)
    b = np.cross(t, n)
    b /= (np.linalg.norm(b) + 1e-12)
    return n, b

def tube_mesh(points, radius, n_circle=16):
    """
    [internal]
    [internal]
    """
    P = points
    R = np.asarray(radius)
    N = len(P)
    if N < 2:
        # [internal note]
        center = P[0]
        theta = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
        phi = np.linspace(0, np.pi, n_circle)
        xv = R[0]*np.outer(np.cos(theta), np.sin(phi))
        yv = R[0]*np.outer(np.sin(theta), np.sin(phi))
        zv = R[0]*np.outer(np.ones_like(theta), np.cos(phi))
        verts = np.c_[xv.ravel()+center[0], yv.ravel()+center[1], zv.ravel()+center[2]]
        faces=[]
        for i in range(n_circle-1):
            for j in range(n_circle-1):
                a=i*n_circle+j; b=a+1; c=a+n_circle; d=c+1
                faces += [(a,b,c),(b,d,c)]
        return verts, np.array(faces, dtype=int)

    # [internal note]
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= (np.linalg.norm(T,axis=1,keepdims=True)+1e-12)

    # [internal note]
    rings = []
    for i in range(N):
        n, b = _frame_from_tangent(T[i])
        thetas = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
        ring = (P[i] + R[i]*np.cos(thetas)[:,None]*n + R[i]*np.sin(thetas)[:,None]*b)
        rings.append(ring)
    rings = np.array(rings)     # (N, n_circle, 3)
    verts = rings.reshape(-1,3)

    # [internal note]
    faces = []
    for i in range(N-1):
        for j in range(n_circle):
            a = i*n_circle + j
            b = i*n_circle + (j+1)%n_circle
            c = (i+1)*n_circle + j
            d = (i+1)*n_circle + (j+1)%n_circle
            faces += [(a,b,c),(b,d,c)]
    faces = np.array(faces, dtype=int)
    return verts, faces

# [internal note]

def centerline_metrics(Pp, Dp, Pg, Dg, n_resample=400):
    """[see source]"""
    # [internal note]
    Pp2, Dp2 = resample_curve(Pp, Dp, n=n_resample)
    Pg2, Dg2 = resample_curve(Pg, Dg, n=n_resample)

    # [internal note]
    len_pred = arclength(Pp2)[-1]
    len_gt   = arclength(Pg2)[-1]
    len_err  = abs(len_pred - len_gt)

    # [internal note]
    # [internal note]
    diam_rmse = np.sqrt(np.mean((Dp2 - Dg2)**2))

    # [internal note]
    tree_g = cKDTree(Pg2)
    tree_p = cKDTree(Pp2)
    d_p_to_g, _ = tree_g.query(Pp2, k=1)
    d_g_to_p, _ = tree_p.query(Pg2, k=1)
    chamfer = 0.5*(np.mean(d_p_to_g**2) + np.mean(d_g_to_p**2))**0.5  # [internal]
    offset_mean = 0.5*(np.mean(d_p_to_g) + np.mean(d_g_to_p))
    offset_p95  = np.percentile(np.concatenate([d_p_to_g, d_g_to_p]), 95)

    # [internal note]
    # [internal note]
    Rg = Dg2/2.0
    Rp = Dp2/2.0
    # [internal note]
    _, idxg = tree_g.query(Pp2, k=1)
    in_gt = (np.linalg.norm(Pp2 - Pg2[idxg], axis=1) <= Rg[idxg] + 1e-6)
    frac_pred_in_gt = in_gt.mean()

    # [internal note]
    _, idxp = tree_p.query(Pg2, k=1)
    in_pred = (np.linalg.norm(Pg2 - Pp2[idxp], axis=1) <= Rp[idxp] + 1e-6)
    frac_gt_in_pred = in_pred.mean()

    return {
        "length_mm_pred": float(len_pred),
        "length_mm_gt": float(len_gt),
        "length_abs_error_mm": float(len_err),
        "diameter_RMSE_mm": float(diam_rmse),
        "offset_mean_mm": float(offset_mean),
        "offset_p95_mm": float(offset_p95),
        "chamfer_mm": float(chamfer),
        "centerline_overlap_pred_in_gt": float(frac_pred_in_gt),
        "centerline_overlap_gt_in_pred": float(frac_gt_in_pred),
    }

def voxelize_tube(points, diameters, grid=128, margin_mm=5.0):
    """
    [internal]
    [internal]
    """
    P = np.asarray(points); D = np.asarray(diameters)
    R = D/2.0
    # [internal note]
    mins = P.min(0) - (R.max()+margin_mm)
    maxs = P.max(0) + (R.max()+margin_mm)
    box = maxs - mins
    step = box.max() / grid           # [internal]
    nx = int(np.ceil(box[0]/step))+1
    ny = int(np.ceil(box[1]/step))+1
    nz = int(np.ceil(box[2]/step))+1

    xs = mins[0] + np.arange(nx)*step
    ys = mins[1] + np.arange(ny)*step
    zs = mins[2] + np.arange(nz)*step

    tree = cKDTree(P)
    mask = np.zeros((nz, ny, nx), dtype=bool)

    # [internal note]
    for iz, z in enumerate(zs):
        # [internal note]
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        pts = np.stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), z)], axis=1)
        dist, idx = tree.query(pts, k=1)
        inside = dist <= (R[idx] + 1e-8)
        mask[iz] = inside.reshape(ny, nx)
    spacing = (step, step, step)   # (dz, dy, dx) mm
    origin = mins
    return mask, spacing, origin

def dice_hd95_from_vox(mask_pred, mask_gt, spacing):
    """
    Dice / IoU / HD95 [internal]
    Dice = 2|A∩B| / (|A|+|B|)
    HD95：[internal]
    """
    A = mask_pred.astype(bool)
    B = mask_gt.astype(bool)
    inter = np.logical_and(A,B).sum()
    volA = A.sum(); volB = B.sum()
    dice = 2.0*inter / (volA + volB + 1e-8)
    iou  = inter / (volA + volB - inter + 1e-8)

    # HD95
    # [internal note]
    def surface(mask):
        if mask.sum() == 0:
            return mask
        er = binary_erosion(mask, structure=np.ones((3,3,3)), border_value=0)
        return np.logical_and(mask, np.logical_not(er))

    surfA = surface(A)
    surfB = surface(B)
    if surfA.sum()==0 or surfB.sum()==0:
        hd95 = float('inf')
    else:
        # [internal note]
        dtA = distance_transform_edt(~A, sampling=spacing)
        dtB = distance_transform_edt(~B, sampling=spacing)
        da = dtB[surfA]  # A[internal]
        db = dtA[surfB]  # B[internal]
        hd95 = float(np.percentile(np.concatenate([da,db]), 95))

    return float(dice), float(iou), float(hd95)

# [internal note]

def add_tube(fig, points, diam, color, name, opacity=0.55, n_circle=18, show_centerline=False):
    verts, faces = tube_mesh(points, diam/2.0, n_circle=n_circle)
    i,j,k = faces.T
    fig.add_trace(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.8),
        flatshading=False, hoverinfo='skip'
    ))
    if show_centerline:
        fig.add_trace(go.Scatter3d(
            x=points[:,0], y=points[:,1], z=points[:,2],
            mode='lines', name=f'{name}-centerline',
            line=dict(width=3, color=color, dash='dash'),
            hoverinfo='skip'
        ))

def make_figure(segments_pred, segments_gt, title='Vessel Prediction vs Ground Truth'):
    fig = go.Figure()
    # [internal note]
    for (pp,dp,sid) in segments_pred:
        add_tube(fig, pp, dp, color="#1f77b4", name=f'Pred seg{sid}', opacity=0.55, show_centerline=False)
    for (pg,dg,sid) in segments_gt:
        add_tube(fig, pg, dg, color="#d62728", name=f'GT seg{sid}', opacity=0.40, show_centerline=False)

    # [internal note]
    all_pts = np.vstack([s[0] for s in segments_pred] + [s[0] for s in segments_gt])
    mins = all_pts.min(0); maxs = all_pts.max(0)
    rang = (maxs - mins).max()
    mid = (maxs + mins)/2
    axrange = [[mid[i]-rang/2, mid[i]+rang/2] for i in range(3)]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (mm)', range=axrange[0], backgroundcolor="rgb(245,245,245)"),
            yaxis=dict(title='Y (mm)', range=axrange[1], backgroundcolor="rgb(245,245,245)"),
            zaxis=dict(title='Z (mm)', range=axrange[2], backgroundcolor="rgb(245,245,245)"),
            aspectmode='cube'
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0,r=0,t=40,b=0)
    )
    return fig

# [internal note]

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    Pp, Dp, segp = load_path(args.pred)
    Pg, Dg, segg = load_path(args.gt)

    segs_pred = split_by_seg(Pp, Dp, segp)
    segs_gt   = split_by_seg(Pg, Dg, segg)

    # [internal note]
    fig = make_figure(segs_pred, segs_gt, title=args.title)
    html_path = os.path.join(args.outdir, 'figure.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"[see source]")

    # [internal note]
    metrics_list = []
    for (pp,dp,sid) in segs_pred:
        # [internal note]
        gt_match = [t for t in segs_gt if t[2]==sid]
        if len(gt_match)==0:
            # [internal note]
            pg = np.vstack([t[0] for t in segs_gt])
            dg = np.concatenate([t[1] for t in segs_gt])
            gid = -1
        else:
            pg, dg, gid = gt_match[0]

        # [internal note]
        m = centerline_metrics(pp, dp, pg, dg, n_resample=args.resample_n)
        m['seg_id'] = int(sid)
        metrics_list.append(m)

        # [internal note]
        if args.grid > 0:
            mask_p, spacing, _ = voxelize_tube(pp, dp, grid=args.grid, margin_mm=args.margin)
            mask_g, _, _       = voxelize_tube(pg, dg, grid=args.grid, margin_mm=args.margin)
            dice, iou, hd95 = dice_hd95_from_vox(mask_p, mask_g, spacing)
            m['dice_voxel'] = dice
            m['iou_voxel']  = iou
            m['hd95_mm']    = hd95

    # [internal note]
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(args.outdir,'metrics.csv'), index=False)
    agg = {
        "num_segments": int(len(metrics_list)),
        "length_mm_pred_total": float(sum(centerline_metrics(t[0],t[1],t[0],t[1])['length_mm_pred'] for t in segs_pred)),
        "length_mm_gt_total": float(sum(centerline_metrics(t[0],t[1],t[0],t[1])['length_mm_pred'] for t in segs_gt)),
    }
    for k in ['length_abs_error_mm','diameter_RMSE_mm','offset_mean_mm','offset_p95_mm','chamfer_mm',
              'centerline_overlap_pred_in_gt','centerline_overlap_gt_in_pred',
              'dice_voxel','iou_voxel','hd95_mm']:
        if k in df.columns:
            agg['mean_'+k] = float(df[k].mean())
            agg['median_'+k] = float(df[k].median())

    # [internal note]
    with open(os.path.join(args.outdir,'metrics.json'),'w',encoding='utf-8') as f:
        json.dump({"per_segment": metrics_list, "summary": agg}, f, ensure_ascii=False, indent=2)

    print("[see source]")
    print(df.describe(include='all').to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[see source]")
    parser.add_argument("--pred", required=True, help="[see source]")
    parser.add_argument("--gt",   required=True, help="[see source]")
    parser.add_argument("--outdir", default="outputs/viz_case", help="[see source]")
    parser.add_argument("--title", default="Vessel Prediction vs Ground Truth")
    parser.add_argument("--grid", type=int, default=128, help="[see source]")
    parser.add_argument("--margin", type=float, default=5.0, help="[see source]")
    parser.add_argument("--resample_n", type=int, default=400, help="[see source]")
    args = parser.parse_args()
    main(args)
