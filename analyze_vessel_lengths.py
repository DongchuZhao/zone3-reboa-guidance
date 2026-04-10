#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vessel segment length evaluation: predicted vs. ground-truth.

Update highlights (2025-11-10):
- [internal]
- [internal]
- [internal]

[internal]
  # [internal note]
  python analyze_vessel_lengths.py \
    --pred_file "./outputs/predictions/<PID>/lengths.csv" \
    --gt_root "/path/to/data" \
    --out_dir ".\\vessel_length_eval_outputs"

  # [internal note]
  python analyze_vessel_lengths.py \
    --pred_root "./outputs/predictions" \
    --gt_root "/path/to/data" \
    --out_dir ".\\vessel_length_eval_outputs"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------
# Globals / constants
# ------------------------------
SEGMENTS = ["1", "2a", "2b", "3L", "3R", "4L", "4R"]
GT_FILES = {
    "1": "Table_1.csv",
    "2a": "Table_2a.csv",
    "2b": "Table_2b.csv",
    "3L": "Table_3L.csv",
    "3R": "Table_3R.csv",
    "4L": "Table_4L.csv",
    "4R": "Table_4R.csv",
}

SEGMENT_SYNONYMS = {
    "1": ["1", "seg1", "table_1", "t1", "zone1", "zone_1"],
    "2a": ["2a", "seg2a", "table_2a", "t2a", "zone2a", "zone_2a"],
    "2b": ["2b", "seg2b", "table_2b", "t2b", "zone2b", "zone_2b"],
    "3L": ["3l", "seg3l", "table_3l", "t3l", "zone3l", "zone_3l", "3-l", "3_left", "3_l"],
    "3R": ["3r", "seg3r", "table_3r", "t3r", "zone3r", "zone_3r", "3-r", "3_right", "3_r"],
    "4L": ["4l", "seg4l", "table_4l", "t4l", "zone4l", "zone_4l", "4-l", "4_left", "4_l"],
    "4R": ["4r", "seg4r", "table_4r", "t4r", "zone4r", "zone_4r", "4-r", "4_right", "4_r"],
    "total": ["total", "sum", "overall", "all", "total_len", "length_total"],
}

# ------------------------------
# Utilities
# ------------------------------
def _normalize_colname(name: str) -> str:
    import re
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name

def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with multiple encoding fallbacks."""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with encodings {encodings}. Last error: {last_err}")

def _pick_numeric_column(df: pd.DataFrame, prefer_keywords: List[str]) -> Optional[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    lcols = {c: _normalize_colname(c) for c in df.columns}
    for kw in prefer_keywords:
        nkw = _normalize_colname(kw)
        for c, lc in lcols.items():
            if nkw in lc and c in numeric_cols:
                return c
    return numeric_cols[0] if numeric_cols else None

def _coerce_segment_key(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text).strip().lower().replace(" ", "").replace("-", "")
    for seg, synonyms in SEGMENT_SYNONYMS.items():
        if any(t == s.lower() for s in [seg] + synonyms):
            return seg
    import re
    m = re.search(r"(?:table|seg|zone)?([1234])(a|b|l|r)?$", t)
    if m:
        base, suffix = m.group(1), (m.group(2) or "").upper()
        seg = base + suffix
        if seg in SEGMENTS:
            return seg
    return None

def _extract_pred_lengths(df: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if df is None or df.empty:
        return result
    norm_map = {c: _normalize_colname(c) for c in df.columns}
    inv_map = {}
    for c, nc in norm_map.items():
        if nc not in inv_map or len(c) > len(inv_map[nc]):
            inv_map[nc] = c
    seg_col = None
    for key in ["segment", "seg", "name", "part", "[see source]", "[see source]", "[see source]", "[see source]"]:
        nk = _normalize_colname(key)
        if nk in inv_map:
            seg_col = inv_map[nk]
            break
    if seg_col is not None:
        val_col = _pick_numeric_column(df, ["pred", "length", "len", "[see source]", "[see source]", "value"])
        if val_col is None:
            raise ValueError("Cannot find numeric length column in prediction CSV.")
        for _, row in df.iterrows():
            seg_raw = str(row[seg_col]).strip()
            seg_key = _coerce_segment_key(seg_raw)
            if seg_key is None:
                continue
            val = row[val_col]
            if pd.notna(val):
                result[seg_key] = float(val)
    else:
        for seg in SEGMENTS + ["total"]:
            for syn in SEGMENT_SYNONYMS.get(seg, []):
                nsyn = _normalize_colname(syn)
                if nsyn in inv_map:
                    col = inv_map[nsyn]
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not series.empty:
                        result[seg] = float(series.iloc[0])
                    break
    if "total" not in result:
        subtotal = sum(result.get(s, 0.0) for s in SEGMENTS if s in result)
        if subtotal > 0:
            result["total"] = subtotal
    return result

def _extract_gt_length_from_table(path: Path) -> Optional[float]:
    df = _safe_read_csv(path)
    distance_col = None
    for c in df.columns:
        if _normalize_colname(c) in {"distance", "dist", "length", "len"}:
            distance_col = c
            break
    if distance_col is None:
        distance_col = df.columns[0]
    series = pd.to_numeric(df[distance_col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])

def _extract_gt_lengths(gt_tables_dir: Path) -> Dict[str, float]:
    gt: Dict[str, float] = {}
    for seg, fname in GT_FILES.items():
        fpath = gt_tables_dir / fname
        if fpath.exists():
            val = _extract_gt_length_from_table(fpath)
            if val is not None:
                gt[seg] = val
    if gt:
        gt["total"] = sum(gt.get(s, 0.0) for s in SEGMENTS if s in gt)
    return gt

def _setup_matplotlib():
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def _find_patients_from_root(pred_root: Path) -> List[Tuple[str, Path]]:
    files = list(pred_root.rglob("lengths.csv"))
    pairs = []
    for f in files:
        pid = f.parent.name
        import re
        if re.match(r"^S\d{4,5}$", pid, flags=re.IGNORECASE):
            pairs.append((pid.upper(), f))
        else:
            m = re.match(r"^(S\d{4,5})", pid, flags=re.IGNORECASE)
            if m:
                pairs.append((m.group(1).upper(), f))
    best: Dict[str, Path] = {}
    for pid, p in pairs:
        if pid not in best or len(str(p).split("/")) < len(str(best[pid]).split("/")):
            best[pid] = p
    return sorted(best.items(), key=lambda x: x[0])

def _patient_from_pred_file(pred_file: Path) -> str:
    """Infer patient ID from the parent folder name like S5360."""
    pid = pred_file.parent.name.upper()
    return pid

def _infer_gt_tables_dir(gt_root: Path, patient_id: str) -> Optional[Path]:
    candidates = [
        gt_root / patient_id / patient_id / "Tables",
        gt_root / patient_id / "Tables",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(parents=True, exist_ok=True)
    return p

# ------------------------------
# Main analysis
# ------------------------------
def analyze(pred_files: List[Tuple[str, Path]], gt_root: Path, out_dir: Path) -> None:
    _setup_matplotlib()
    out_dir = _ensure_outdir(out_dir)

    rows = []
    total_rows = []
    missing = []

    for pid, pred_csv in pred_files:
        gt_tables_dir = _infer_gt_tables_dir(gt_root, pid)
        if gt_tables_dir is None:
            missing.append((pid, "GT Tables dir not found"))
            continue
        try:
            pred_df = _safe_read_csv(pred_csv)
            pred_map = _extract_pred_lengths(pred_df)
        except Exception as e:
            missing.append((pid, f"Read pred fail: {e}"))
            continue
        gt_map = _extract_gt_lengths(gt_tables_dir)
        if not gt_map:
            missing.append((pid, "No GT segment values found"))
            continue

        segs_present = sorted(set(list(SEGMENTS) + ["total"]))
        for seg in segs_present:
            gt_val = gt_map.get(seg, np.nan)
            pred_val = pred_map.get(seg, np.nan)
            if pd.isna(gt_val) and pd.isna(pred_val):
                continue
            ae = np.nan
            re = np.nan
            if not pd.isna(gt_val) and not pd.isna(pred_val):
                ae = abs(pred_val - gt_val)
                re = abs(pred_val - gt_val) / gt_val * 100.0 if gt_val != 0 else np.nan
            if seg == "total":
                total_rows.append({
                    "patient_id": pid,
                    "pred_total_mm": pred_val,
                    "gt_total_mm": gt_val,
                    "AE_mm": ae,
                    "RE_percent": re,
                })
            else:
                rows.append({
                    "patient_id": pid,
                    "segment": seg,
                    "pred_mm": pred_val,
                    "gt_mm": gt_val,
                    "AE_mm": ae,
                    "RE_percent": re,
                })

    df_segments = pd.DataFrame(rows)
    df_total = pd.DataFrame(total_rows)

    seg_csv = out_dir / "segment_length_errors.csv"
    total_csv = out_dir / "total_length_errors.csv"
    df_segments.to_csv(seg_csv, index=False, encoding="utf-8-sig")
    df_total.to_csv(total_csv, index=False, encoding="utf-8-sig")

    try:
        xlsx_path = out_dir / "length_errors_summary.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_segments.to_excel(writer, index=False, sheet_name="per_segment")
            df_total.to_excel(writer, index=False, sheet_name="per_patient_total")
        print(f"[OK] Saved Excel: {xlsx_path}")
    except Exception as e:
        print(f"[WARN] Could not write Excel (openpyxl missing?): {e}")

    if missing:
        miss_path = out_dir / "missing_or_failed.txt"
        with miss_path.open("w", encoding="utf-8") as f:
            for pid, reason in missing:
                f.write(f"{pid}\t{reason}\n")
        print(f"[INFO] Some patients skipped, see: {miss_path}")

    # ------------------------------
    # Figures
    # ------------------------------
    def _scatter_pred_vs_gt(sub_df: pd.DataFrame, title: str, out_name: str):
        sub = sub_df.dropna(subset=["pred", "gt"])
        if sub.empty:
            return
        x = sub["gt"].to_numpy(dtype=float)
        y = sub["pred"].to_numpy(dtype=float)

        min_v = float(np.nanmin(np.concatenate([x, y])))
        max_v = float(np.nanmax(np.concatenate([x, y])))
        pad = 0.05 * (max_v - min_v if max_v > min_v else 1.0)
        lo, hi = min_v - pad, max_v + pad

        plt.figure(figsize=(6, 6), dpi=150)
        plt.scatter(x, y, s=24, alpha=0.8)
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)  # y = x
        plt.xlabel("[see source]")
        plt.ylabel("[see source]")
        plt.title(title)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / out_name, bbox_inches="tight")
        plt.close()

    for seg in SEGMENTS:
        df_seg = df_segments[df_segments["segment"] == seg].copy()
        if df_seg.empty:
            continue
        _scatter_pred_vs_gt(
            sub_df=df_seg.rename(columns={"pred_mm": "pred", "gt_mm": "gt"}),
            title=f"[see source]",
            out_name=f"scatter_segment_{seg}.png",
        )

    _scatter_pred_vs_gt(
        sub_df=df_total.rename(columns={"pred_total_mm": "pred", "gt_total_mm": "gt"}),
        title="[see source]",
        out_name=f"scatter_total.png",
    )

    if not df_segments.empty:
        plt.figure(figsize=(8, 5), dpi=150)
        plot_df = df_segments.copy()
        plot_df["segment"] = pd.Categorical(plot_df["segment"], categories=SEGMENTS, ordered=True)
        plot_df = plot_df.sort_values("segment")

        data = [plot_df.loc[plot_df["segment"] == s, "AE_mm"].dropna().to_numpy() for s in SEGMENTS]
        # Matplotlib 3.9+: use tick_labels instead of deprecated labels
        plt.boxplot(data, tick_labels=SEGMENTS, showmeans=True)
        plt.xlabel("[see source]")
        plt.ylabel("[see source]")
        plt.title("[see source]")
        plt.grid(True, linestyle=":", linewidth=0.6, axis="y", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "boxplot_AE_by_segment.png", bbox_inches="tight")
        plt.close()

        means = [np.nanmean(d) if len(d) else np.nan for d in data]
        stds  = [np.nanstd(d) if len(d) else np.nan for d in data]
        import numpy as np
        x = np.arange(len(SEGMENTS))

        plt.figure(figsize=(8, 5), dpi=150)
        plt.bar(x, means, yerr=stds, capsize=3)
        plt.xticks(x, SEGMENTS)
        plt.xlabel("[see source]")
        plt.ylabel("[see source]")
        plt.title("[see source]")
        plt.grid(True, linestyle=":", linewidth=0.6, axis="y", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "bar_mean_AE_by_segment.png", bbox_inches="tight")
        plt.close()

        re_data  = [plot_df.loc[plot_df["segment"] == s, "RE_percent"].dropna().to_numpy() for s in SEGMENTS]
        re_means = [np.nanmean(d) if len(d) else np.nan for d in re_data]
        re_stds  = [np.nanstd(d) if len(d) else np.nan for d in re_data]
        x = np.arange(len(SEGMENTS))

        plt.figure(figsize=(8, 5), dpi=150)
        plt.bar(x, re_means, yerr=re_stds, capsize=3)
        plt.xticks(x, SEGMENTS)
        plt.xlabel("[see source]")
        plt.ylabel("[see source]")
        plt.title("[see source]")
        plt.grid(True, linestyle=":", linewidth=0.6, axis="y", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "bar_mean_RE_by_segment.png", bbox_inches="tight")
        plt.close()

    n_patients = len(set(df_segments["patient_id"])) if not df_segments.empty else len(set(df_total["patient_id"]))
    print(f"[see source]")
    print(f"[see source]")
    print(f"[see source]")
    print(f"[see source]")

# ------------------------------
# CLI
# ------------------------------
def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Compare predicted vessel segment lengths to ground truth; output tables and plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default=r"H:\PyCharm\Projects\BloodVesselPrediction_rich2D\BloodVesselPrediction_rich2D\outputs\predictions",
        help="Root directory that contains per-patient folders with lengths.csv (ignored if --pred_file is provided)",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default=None,
        help="Optional single predictions CSV (lengths.csv) for one patient, e.g., ...\\S5360\\lengths.csv",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default=r"E:\ABD1_300",
        help="Ground-truth root directory that contains patient folders with Tables subfolder",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, a folder 'vessel_length_eval_outputs' will be created in the current directory.",
    )
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    gt_root = Path(args.gt_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd() / "vessel_length_eval_outputs"
    out_dir = out_dir.resolve()

    if args.pred_file:
        pred_file = Path(args.pred_file).resolve()
        print(f"[CFG] pred_file = {pred_file}")
        print(f"[CFG] gt_root   = {gt_root}")
        print(f"[CFG] out_dir   = {out_dir}")
        if not pred_file.exists():
            print(f"[ERROR] Predictions file does not exist: {pred_file}")
            sys.exit(1)
        pid = _patient_from_pred_file(pred_file)
        pred_files = [(pid, pred_file)]
    else:
        pred_root = Path(args.pred_root).resolve()
        print(f"[CFG] pred_root = {pred_root}")
        print(f"[CFG] gt_root   = {gt_root}")
        print(f"[CFG] out_dir   = {out_dir}")
        if not pred_root.exists():
            print(f"[ERROR] Predictions root does not exist: {pred_root}")
            sys.exit(1)
        pred_files = _find_patients_from_root(pred_root)

    if not gt_root.exists():
        print(f"[ERROR] Ground-truth root does not exist: {gt_root}")
        sys.exit(1)

    if not pred_files:
        print(f"[WARN] No predictions found.")
        sys.exit(0)

    analyze(pred_files, gt_root, out_dir)

if __name__ == "__main__":
    main()
