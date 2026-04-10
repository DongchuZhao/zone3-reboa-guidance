# utils/discovery.py
import os, re, glob
from typing import List, Tuple, Dict

PID_PAT = re.compile(r"^(S|SCT|ASCT)\d+$")  # [internal]

def _looks_like_pid(name: str) -> bool:
    return PID_PAT.match(name) is not None

def _first_nrrd_file(origin_dir: str) -> str | None:
    """
    [internal]
    """
    if not os.path.isdir(origin_dir):
        return None
    # [internal note]
    cands = glob.glob(os.path.join(origin_dir, "*.nrrd"))
    return cands[0] if cands else None

def _check_tables(tables_dir: str, segment_names: List[str]) -> Tuple[bool, List[str]]:
    """
    [internal]
    """
    missing = []
    for fn in (segment_names or []):
        if not os.path.isfile(os.path.join(tables_dir, fn)):
            missing.append(fn)
    return (len(missing) == 0), missing

def _choose_patient_home(data_root: str, pid: str) -> str | None:
    """
    [internal]
    [internal]
    """
    # [internal note]
    a = os.path.join(data_root, pid, pid)
    # [internal note]
    b = os.path.join(data_root, pid)

    if os.path.isdir(a):
        return a
    if os.path.isdir(b):
        return b
    return None

def discover_patients(
    data_root: str,
    origin_subdir: str,
    models_subdir: str,
    tables_subdir: str,
    ct_filename: str | None = None,               # [internal]
    surface_seg_filename: str | None = None,      # [internal]
    segment_names: List[str] | None = None,
    report_path: str = "outputs/logs/discovery_report.txt"
) -> List[str]:
    """
    [internal]
      1) [internal]
      2) [internal]
      3) [internal]
    [internal]
    """

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    valid: List[str] = []
    reasons: Dict[str, List[str]] = {}

    if not os.path.isdir(data_root):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"[see source]")
        return valid

    # [internal note]
    top_dirs = [d for d in os.listdir(data_root) if _looks_like_pid(d) and os.path.isdir(os.path.join(data_root, d))]
    top_dirs.sort()

    for pid in top_dirs:
        home = _choose_patient_home(data_root, pid)
        if home is None:
            reasons.setdefault(pid, []).append("[see source]")
            continue

        origin_dir = os.path.join(home, origin_subdir)
        tables_dir = os.path.join(home, tables_subdir)
        # [internal note]

        # [internal note]
        ct_path = _first_nrrd_file(origin_dir)
        if ct_path is None:
            reasons.setdefault(pid, []).append(f"[see source]")

        # [internal note]
        ok_tables, missing = _check_tables(tables_dir, segment_names or [])
        if not ok_tables:
            reasons.setdefault(pid, []).append("[see source]" + ", ".join(missing))

        # [internal note]
        if surface_seg_filename:
            ss = os.path.join(home, models_subdir, surface_seg_filename)
            if not os.path.isfile(ss):
                # [internal note]
                reasons.setdefault(pid, []).append(f"[see source]")

        if ct_path is not None and ok_tables:
            valid.append(pid)

    # [internal note]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"[SUMMARY] data_root = {data_root}\n")
        f.write(f"[see source]")
        f.write(f"[see source]")
        f.write(f"[see source]")
        if reasons:
            f.write("[see source]")
            for pid in sorted(reasons.keys()):
                f.write(f"{pid}: " + " | ".join(reasons[pid]) + "\n")

    return valid
