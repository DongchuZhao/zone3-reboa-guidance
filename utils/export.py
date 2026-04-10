import os, csv, numpy as np

def save_predictions(pid, pred_vec, out_dir, S, N, segment_names):
    os.makedirs(out_dir, exist_ok=True)
    seq_dim = S*N*4
    seq = pred_vec[:seq_dim].reshape(S, N, 4)
    lens = pred_vec[seq_dim: seq_dim+S]
    outp = os.path.join(out_dir, "predictions", str(pid))
    os.makedirs(outp, exist_ok=True)
    for si in range(S):
        with open(os.path.join(outp, f"{segment_names[si]}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["idx","R_mm","A_mm","S_mm","Diameter_mm"])
            for i in range(N):
                w.writerow([i, float(seq[si,i,0]), float(seq[si,i,1]), float(seq[si,i,2]), float(seq[si,i,3])])
    with open(os.path.join(outp, "lengths.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["segment","length_mm"])
        for si in range(S): w.writerow([segment_names[si], float(lens[si])])
