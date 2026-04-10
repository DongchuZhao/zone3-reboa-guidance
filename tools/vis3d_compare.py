import os, numpy as np
import plotly.graph_objects as go

# [internal note]
PID = "SXXXX"
gt_path   = f"outputs/predict/{PID}/gt_centerline.npy"    # [N,3], [internal]
pred_path = f"outputs/predict/{PID}/pred_centerline.npy"  # [M,3], [internal]

gt   = np.load(gt_path)   # shape [N,3]
pred = np.load(pred_path) # shape [M,3]

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=gt[:,0], y=gt[:,1], z=gt[:,2],
    mode="lines", name="GT", line=dict(width=8)))
fig.add_trace(go.Scatter3d(
    x=pred[:,0], y=pred[:,1], z=pred[:,2],
    mode="lines", name="Pred", line=dict(width=6)))

fig.update_layout(
    title=f"Vessel Centerline Compare · {PID}",
    scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
os.makedirs(f"outputs/vis/{PID}", exist_ok=True)
out_html = f"outputs/vis/{PID}/compare_centerline.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print("Saved:", out_html)
