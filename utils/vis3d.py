# utils/vis3d.py
import numpy as np, torch, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def _to_np(x):
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def render_3d_static(pred_coords, gt_coords, pred_diam, gt_diam, out_png):
    """
    [internal]
    """
    pc = _to_np(pred_coords)  # (S,N,3)
    gc = _to_np(gt_coords)
    pd = _to_np(pred_diam)    # (S,N)
    gd = _to_np(gt_diam)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    for s in range(pc.shape[0]):
        ax.plot(pc[s,:,0], pc[s,:,1], pc[s,:,2], lw=np.clip(pd[s]/4.0, 1.0, 6.0), alpha=0.9, color="#1f77b4")
        ax.plot(gc[s,:,0], gc[s,:,1], gc[s,:,2], lw=np.clip(gd[s]/4.0, 1.0, 6.0), alpha=0.6, color="#d62728", linestyle="--")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("3D centerlines: pred (blue) vs GT (red dashed)")
    ax.view_init(elev=20, azim=35)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)

def render_3d_interactive(pred_coords, gt_coords, pred_diam, gt_diam, out_html):
    """
    [internal]
    """
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plotly_plot
    except Exception:
        return
    pc = _to_np(pred_coords); gc = _to_np(gt_coords)
    pd = _to_np(pred_diam);   gd = _to_np(gt_diam)

    traces = []
    for s in range(pc.shape[0]):
        traces.append(go.Scatter3d(x=pc[s,:,0], y=pc[s,:,1], z=pc[s,:,2],
            mode="lines", line=dict(width=float(np.clip(pd[s].mean()/2.0,2,8))), name=f"pred seg{s}"))
        traces.append(go.Scatter3d(x=gc[s,:,0], y=gc[s,:,1], z=gc[s,:,2],
            mode="lines", line=dict(width=float(np.clip(gd[s].mean()/2.0,2,8)), dash="dash"),
            name=f"gt seg{s}", opacity=0.7))
    layout = go.Layout(scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
                       title="Centerlines (interactive)", showlegend=True)
    fig = go.Figure(data=traces, layout=layout)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    plotly_plot(fig, filename=out_html, auto_open=False)
