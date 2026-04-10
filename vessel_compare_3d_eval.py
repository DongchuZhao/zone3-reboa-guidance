# -*- coding: utf-8 -*-
"""
vessel_compare_3d_eval_final_7pts.py

[internal]
1. [internal]
2. [internal]
3. [internal]
4. [internal]
5. [internal]

[internal]
- [internal]

[internal]
    [internal]
"""

import os
import sys
import math
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# [internal note]

# [internal note]
PRED_ROOT = r"H:\PyCharm\Projects\BloodVesselPrediction_rich2D\BloodVesselPrediction_rich2D\outputs\predictions"
# [internal note]
GT_ROOT = r"E:\ABD1_450"
# [internal note]
OUT_ROOT = r"H:\PyCharm\Projects\BloodVesselPrediction_rich2D\BloodVesselPrediction_rich2D\outputs\eval_zoneIII_7pts"


ANGLE_DEG = 45.0
N_PUNCTURE_POINTS = 5
BALLOON_LEN_MM = 40.0

# [internal note]
SEG_TABLES = [
    "Table_1.csv", "Table_2a.csv", "Table_2b.csv",
    "Table_3L.csv", "Table_3R.csv", "Table_4L.csv", "Table_4R.csv"
]


# [internal note]

def _norm_col(s: str) -> str:
    """[see source]"""
    if s is None:
        return ""
    s = str(s).strip().lower()
    for ch in ["(", ")", "[", "]", "{", "}", " "]:
        s = s.replace(ch, "")
    s = s.replace("__", "_").replace("-", "").replace("/", "")
    return s


def _read_csv_smart(path: Path) -> pd.DataFrame:
    """[see source]"""
    last_err = None
    for enc in ["utf-8-sig", "gbk", "utf-8"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _find_files(folder: Path, patterns: List[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        found.extend([Path(p) for p in glob.glob(str(folder / pat))])
    return sorted(set(found))


def _segment_name_from_filename(fn: str) -> str:
    base = os.path.basename(fn)
    while base.lower().endswith(".csv"):
        base = base[:-4]
    if base.lower().startswith("table_"):
        return base[6:]
    return base


def _to_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _pick_cols(df: pd.DataFrame) -> Dict[str, str]:
    """[see source]"""
    inv = {}
    for c in df.columns:
        nc = _norm_col(c)
        if nc not in inv:
            inv[nc] = c

    def choose(cands: List[str]) -> str:
        for cc in cands:
            ncc = _norm_col(cc)
            if ncc in inv:
                return inv[ncc]
        return ""

    return {
        "x": choose(["RAS_R", "R_mm", "R", "X"]),
        "y": choose(["RAS_A", "A_mm", "A", "Y"]),
        "z": choose(["RAS_S", "S_mm", "S", "Z"]),
        "d": choose(["Diameter (CE)", "Diameter_mm", "Diameter", "Diam", "D"]),
        "i": choose(["idx", "index", "id", "pointid"]),
        "abs": choose(["abs_distance_mm", "absdistancemm", "absdistance", "distance_to_skin_mm"]),
    }


def _compute_cumlen(points: np.ndarray) -> Tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.zeros((len(points),), dtype=float), 0.0
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, float(cum[-1])


def _choose_candidate_indices(n: int, k: int = 5) -> List[int]:
    if n <= 2:
        return []
    k = min(k, n - 2)
    idxs = sorted(set(int(round(x)) for x in np.linspace(1, n - 2, k)))
    return idxs


def _flip_segment(seg: dict) -> dict:
    out = dict(seg)
    out["points"] = seg["points"][::-1].copy()
    out["radii"] = seg["radii"][::-1].copy()
    if seg.get("abs_distance") is not None:
        out["abs_distance"] = seg["abs_distance"][::-1].copy()
    out["cumlen"], out["length_mm"] = _compute_cumlen(out["points"])
    return out


def _orient_to_connect(seg: dict, connect_point: np.ndarray, connect_at_end: bool) -> dict:
    P = seg["points"]
    d0 = float(np.linalg.norm(P[0] - connect_point))
    d1 = float(np.linalg.norm(P[-1] - connect_point))
    if connect_at_end:
        return seg if d1 <= d0 else _flip_segment(seg)
    else:
        return seg if d0 <= d1 else _flip_segment(seg)


def _orient_2b_2a(gt: Dict[str, dict]) -> Tuple[dict, dict]:
    seg2b = gt["2b"]
    seg2a = gt["2a"]
    best = None
    for a2b in [seg2b, _flip_segment(seg2b)]:
        for a2a in [seg2a, _flip_segment(seg2a)]:
            dist = float(np.linalg.norm(a2b["points"][-1] - a2a["points"][0]))
            if best is None or dist < best[0]:
                best = (dist, a2b, a2a)
    return best[1], best[2]


# [internal note]

def _load_gt_segments(gt_tables_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for fname in SEG_TABLES:
        candidates = _find_files(gt_tables_dir, [fname, fname + ".csv"])
        if not candidates:
            continue
        path = candidates[0]
        try:
            df = _read_csv_smart(path)
        except Exception:
            continue

        cols = _pick_cols(df)
        if not (cols["x"] and cols["y"] and cols["z"] and cols["d"]):
            continue

        if cols["i"]:
            try:
                df = df.sort_values(by=cols["i"], kind="mergesort")
            except Exception:
                pass

        X = _to_float_array(df[cols["x"]])
        Y = _to_float_array(df[cols["y"]])
        Z = _to_float_array(df[cols["z"]])
        D = _to_float_array(df[cols["d"]])
        R = np.maximum(D * 0.5, 0.1)

        P = np.stack([X, Y, Z], axis=1)
        mask = np.isfinite(P).all(axis=1) & np.isfinite(R)
        P = P[mask]
        R = R[mask]

        absd = None
        if cols["abs"]:
            absd = _to_float_array(df[cols["abs"]])[mask]

        if len(P) < 2:
            continue

        cum, tot = _compute_cumlen(P)
        seg = _segment_name_from_filename(path.name)
        out[seg] = dict(points=P, radii=R, abs_distance=absd, cumlen=cum, length_mm=tot, path=str(path))
    return out


def _load_pred_lengths(pred_dir: Path) -> Dict[str, float]:
    candidates = _find_files(pred_dir, ["lengths.csv", "Lengths.csv", "*lengths*.csv"])
    if not candidates:
        raise FileNotFoundError(f"[see source]")
    path = candidates[0]
    df = _read_csv_smart(path)

    cols_norm = {_norm_col(c): c for c in df.columns}
    seg_col = cols_norm.get("segment") or cols_norm.get("seg") or list(df.columns)[0]
    len_col = cols_norm.get("length_mm") or cols_norm.get("length") or cols_norm.get("len")

    if len_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c != seg_col:
                len_col = c
                break

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        seg = str(row[seg_col]).strip().replace("\\", "/").split("/")[-1]
        if seg.lower().startswith("table_"): seg = seg[6:]
        seg = seg.replace(".csv", "").strip()
        try:
            val = float(row[len_col])
            out[seg] = val
        except Exception:
            continue

    out["_file"] = str(path)
    return out


# [internal note]

def _eval_side_zoneIII_v2(side: str, gt_raw: Dict[str, dict], pred_len: Dict[str, float],
                          angle_deg: float, n_points: int) -> Tuple[pd.DataFrame, dict, dict]:
    """
    [internal]
    [internal]
    - Success: 40mm [internal]
    - Too High: [internal]
    - Too Low: [internal]
    """
    seg4 = f"4{side}"
    seg3 = f"3{side}"

    # [internal note]
    o2b, o2a = _orient_2b_2a(gt_raw)
    p2b_start = o2b["points"][0]  # 2b Distal ([internal]
    o3 = _orient_to_connect(gt_raw[seg3], p2b_start, connect_at_end=True)
    o4 = _orient_to_connect(gt_raw[seg4], o3["points"][0], connect_at_end=True)

    if o4.get("abs_distance") is None:
        raise ValueError(f"[see source]")

    # [internal note]
    len2a_gt = float(o2a["length_mm"])
    len2b_gt = float(o2b["length_mm"])
    len3_gt = float(o3["length_mm"])
    len4_gt = float(o4["length_mm"])

    # [internal note]
    len2a_pr = float(pred_len.get("2a", 0))
    len2b_pr = float(pred_len.get("2b", 0))
    len3_pr = float(pred_len.get(seg3, 0))
    len4_pr = float(pred_len.get(seg4, 0))

    idxs = _choose_candidate_indices(len(o4["points"]), n_points)
    ang = math.radians(angle_deg)
    cosang = math.cos(ang)

    rows = []
    half_balloon = BALLOON_LEN_MM / 2.0

    for i in idxs:
        absd = float(o4["abs_distance"][i])
        tissue = absd / (cosang + 1e-8)

        # [internal note]
        rem4_gt = len4_gt - float(o4["cumlen"][i])

        # [internal note]
        frac = rem4_gt / len4_gt if len4_gt > 1e-6 else 1.0
        rem4_pr = frac * len4_pr

        # [internal note]
        distal_boundary_true = tissue + rem4_gt + len3_gt  # Bifurcation
        prox_boundary_true = distal_boundary_true + len2b_gt + len2a_gt  # Lowest Renal

        # [internal note]
        distal_boundary_pred = tissue + rem4_pr + len3_pr
        prox_boundary_pred = distal_boundary_pred + len2b_pr + len2a_pr
        target_mid_pred = (distal_boundary_pred + prox_boundary_pred) * 0.5

        # [internal note]
        balloon_distal_pos = target_mid_pred - half_balloon
        balloon_proximal_pos = target_mid_pred + half_balloon

        # [internal note]
        # [internal note]
        is_success = (balloon_distal_pos >= distal_boundary_true) and (balloon_proximal_pos <= prox_boundary_true)

        # [internal note]
        is_too_high = (balloon_proximal_pos > prox_boundary_true)

        # [internal note]
        is_too_low = (balloon_distal_pos < distal_boundary_true)

        # [internal note]
        true_mid = (distal_boundary_true + prox_boundary_true) * 0.5
        mid_error = target_mid_pred - true_mid

        rows.append(dict(
            side=side,
            puncture_index=i,
            abs_distance_mm=absd,
            tissue_len_mm=tissue,

            true_distal_bound=distal_boundary_true,
            true_prox_bound=prox_boundary_true,

            pred_target_mid=target_mid_pred,
            balloon_distal_pos=balloon_distal_pos,
            balloon_proximal_pos=balloon_proximal_pos,

            success=int(is_success),
            too_high=int(is_too_high),
            too_low=int(is_too_low),
            mid_error_mm=mid_error
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid puncture points for {side}")

    # [internal note]
    n_total = len(df)
    n_succ = df["success"].sum()
    n_high = df["too_high"].sum()
    n_low = df["too_low"].sum()

    # [internal note]
    cands = df[df["success"] == 1]
    if cands.empty:
        cands = df
    best_row = cands.loc[cands["mid_error_mm"].abs().idxmin()]

    summary = dict(
        side=side,
        total_points=n_total,
        success_rate=n_succ / n_total,
        too_high_rate=n_high / n_total,
        too_low_rate=n_low / n_total,
        recommended_mid_mm=float(best_row["pred_target_mid"]),
        recommended_idx=int(best_row["puncture_index"]),
        best_mid_error=float(best_row["mid_error_mm"])
    )

    oriented = {"4": o4, "3": o3, "2b": o2b, "2a": o2a}
    return df, summary, oriented


# [internal note]

def _sample_point(P: np.ndarray, s: float) -> np.ndarray:
    """[see source]"""
    if len(P) == 0: return np.zeros(3)
    if s <= 0: return P[0]
    seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    if s >= cum[-1]: return P[-1]
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    t = (s - cum[idx]) / (seglen[idx] + 1e-12)
    return P[idx] * (1 - t) + P[idx + 1] * t


def _build_path_points(o4, idx, o3, o2b, o2a) -> np.ndarray:
    return np.vstack([o4["points"][idx:], o3["points"], o2b["points"], o2a["points"]])


def _plot_zone_eval_3d_v2(oriented: dict, chosen_idx: int,
                          pred_mid_mm: float, side: str, out_html: Path):
    """
    3D [internal]
    """
    o4, o3, o2b, o2a = oriented["4"], oriented["3"], oriented["2b"], oriented["2a"]

    pathP = _build_path_points(o4, chosen_idx, o3, o2b, o2a)

    absd = float(o4["abs_distance"][chosen_idx])
    tissue = absd / (math.cos(math.radians(ANGLE_DEG)) + 1e-8)

    # [internal note]
    rem4 = float(o4["length_mm"] - o4["cumlen"][chosen_idx])
    start_2b = rem4 + float(o3["length_mm"])
    end_2b = start_2b + float(o2b["length_mm"])
    end_2a = end_2b + float(o2a["length_mm"])

    # [internal note]
    pred_mid_vessel = pred_mid_mm - tissue
    balloon_start = pred_mid_vessel - BALLOON_LEN_MM / 2.0
    balloon_end = pred_mid_vessel + BALLOON_LEN_MM / 2.0

    fig = go.Figure()

    # [internal note]
    fig.add_trace(go.Scatter3d(
        x=pathP[:, 0], y=pathP[:, 1], z=pathP[:, 2],
        mode='lines', line=dict(color='lightgray', width=4),
        name='Access Path'
    ))

    # [internal note]
    ss_zone = np.linspace(start_2b, end_2a, 100)
    pts_zone = np.vstack([_sample_point(pathP, s) for s in ss_zone])
    fig.add_trace(go.Scatter3d(
        x=pts_zone[:, 0], y=pts_zone[:, 1], z=pts_zone[:, 2],
        mode='lines', line=dict(color='green', width=12),
        opacity=0.4, name='True Zone III'
    ))

    # [internal note]
    ss_bal = np.linspace(max(0, balloon_start), balloon_end, 50)
    pts_bal = np.vstack([_sample_point(pathP, s) for s in ss_bal])

    fig.add_trace(go.Scatter3d(
        x=pts_bal[:, 0], y=pts_bal[:, 1], z=pts_bal[:, 2],
        mode='lines', line=dict(color='orange', width=15),
        name=f'Pred Balloon ({int(BALLOON_LEN_MM)}mm)'
    ))

    # [internal note]
    p_punc = o4["points"][chosen_idx]
    fig.add_trace(go.Scatter3d(
        x=[p_punc[0]], y=[p_punc[1]], z=[p_punc[2]],
        mode='markers', marker=dict(size=6, color='black'),
        name='Puncture Point'
    ))

    fig.update_layout(
        title=f"{side} Side Result (Orange=Balloon, Green=Zone III)",
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(str(out_html))


def _plot_summary_charts(df_batch: pd.DataFrame, out_dir: Path):
    """[see source]"""
    if df_batch.empty: return

    patients = df_batch["patient"]
    x = np.arange(len(patients))

    # [internal note]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    ax.bar(x - width / 2, df_batch["right_success_rate"], width, label="Right", alpha=0.8)
    ax.bar(x + width / 2, df_batch["left_success_rate"], width, label="Left", alpha=0.8)
    ax.plot(x, df_batch["total_success_rate"], "k-o", linewidth=2, label="Total")

    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Patient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "batch_success_rates.png", dpi=200)
    plt.close(fig)

    # [internal note]
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.25
    ax.bar(x - w, df_batch["total_success_rate"], w, label="Success", color='g', alpha=0.7)
    ax.bar(x, df_batch["total_too_high_rate"], w, label="Too High", color='r', alpha=0.7)
    ax.bar(x + w, df_batch["total_too_low_rate"], w, label="Too Low", color='b', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45)
    ax.set_title("Total Success vs Errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "batch_error_analysis.png", dpi=200)
    plt.close(fig)


# [internal note]

def process_patient(patient: str) -> dict:
    pred_dir = Path(PRED_ROOT) / patient
    gt_dir = Path(GT_ROOT) / patient / patient / "Tables"
    out_dir = Path(OUT_ROOT) / patient
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {patient} ...")

    gt = _load_gt_segments(gt_dir)
    if not gt:
        print(f"[WARN] GT not found for {patient}")
        return {}

    try:
        pred_len = _load_pred_lengths(pred_dir)
    except Exception as e:
        print(f"[WARN] Pred lengths not found for {patient}: {e}")
        return {}

    summaries = {}

    for side in ["R", "L"]:
        try:
            df, summ, ori = _eval_side_zoneIII_v2(side, gt, pred_len, ANGLE_DEG, N_PUNCTURE_POINTS)
            summaries[side] = summ

            # [internal note]
            df.to_csv(out_dir / f"details_{side}.csv", index=False, float_format="%.2f")

            # [internal note]
            _plot_zone_eval_3d_v2(ori, summ["recommended_idx"], summ["recommended_mid_mm"], side,
                                  out_dir / f"3d_viz_{side}.html")

        except Exception as e:
            print(f"  [Error] Side {side}: {e}")

    # [internal note]
    r_res = summaries.get("R", {})
    l_res = summaries.get("L", {})

    def get_val(d, k):
        return d.get(k, 0.0)

    def get_count(d):
        return d.get("total_points", 0)

    nR = get_count(r_res)
    nL = get_count(l_res)
    total_pts = nR + nL

    if total_pts > 0:
        tot_succ = (get_val(r_res, "success_rate") * nR + get_val(l_res, "success_rate") * nL) / total_pts
        tot_high = (get_val(r_res, "too_high_rate") * nR + get_val(l_res, "too_high_rate") * nL) / total_pts
        tot_low = (get_val(r_res, "too_low_rate") * nR + get_val(l_res, "too_low_rate") * nL) / total_pts
    else:
        tot_succ = tot_high = tot_low = 0.0

    row = {
        "patient": patient,
        # [internal note]
        "right_success_rate": get_val(r_res, "success_rate"),
        "left_success_rate": get_val(l_res, "success_rate"),
        "total_success_rate": tot_succ,
        "total_too_high_rate": tot_high,
        "total_too_low_rate": tot_low,
        "right_too_high": get_val(r_res, "too_high_rate"),
        "right_too_low": get_val(r_res, "too_low_rate"),
        "left_too_high": get_val(l_res, "too_high_rate"),
        "left_too_low": get_val(l_res, "too_low_rate"),

        # [internal note]
        "right_rec_len_mm": get_val(r_res, "recommended_mid_mm"),
        "left_rec_len_mm": get_val(l_res, "recommended_mid_mm"),
    }

    pd.DataFrame([row]).to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    print(
        f"  -> Total Success: {tot_succ:.2%}, R-Depth: {row['right_rec_len_mm']:.1f}, L-Depth: {row['left_rec_len_mm']:.1f}")
    return row


def main():
    if len(sys.argv) >= 2:
        inp = " ".join(sys.argv[1:]).strip()
    else:
        print("Input Patient ID or ALL:")
        inp = input().strip()

    if inp.upper() == "ALL":
        root = Path(PRED_ROOT)
        patients = sorted([d.name for d in root.iterdir() if d.is_dir()])
    else:
        patients = [x.strip() for x in inp.replace(",", " ").split() if x.strip()]

    results = []
    for p in patients:
        res = process_patient(p)
        if res: results.append(res)

    if results:
        out_root = Path(OUT_ROOT)
        out_root.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df = df.sort_values("patient")

        # [internal note]
        avg_vals = df.mean(numeric_only=True)
        avg_total_succ = avg_vals["total_success_rate"]
        avg_too_high = avg_vals["total_too_high_rate"]
        avg_too_low = avg_vals["total_too_low_rate"]
        avg_rec_R = avg_vals["right_rec_len_mm"]
        avg_rec_L = avg_vals["left_rec_len_mm"]

        print("\n" + "=" * 50)
        print(" BATCH EVALUATION SUMMARY (N=5 Points)")
        print("=" * 50)

        # print(f" Patients Evaluated:       {len(df)}")
        print(f" Average Total Accuracy:   {avg_total_succ:.2%}")
        print(f" Average Too High Error:   {avg_too_high:.2%}")
        print(f" Average Too Low Error:    {avg_too_low:.2%}")
        print("-" * 50)

        print("=" * 50 + "\n")

        # [internal note]
        _plot_summary_charts(df, out_root)
        print("Batch charts generated.")

        # [internal note]
        avg_row = avg_vals.to_dict()
        avg_row["patient"] = "AVERAGE"

        # [internal note]
        df_final = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        csv_path = out_root / "batch_summary_all.csv"
        df_final.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"Batch summary saved to: {csv_path}")


if __name__ == "__main__":
    main()