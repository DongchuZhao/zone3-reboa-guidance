# main.py
import os, random, yaml, math, time, sys, threading
from datetime import datetime
import pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import VesselSeqSampleLoader
from models.teacher_model import TeacherModel
from models.student_model import StudentModel

# [internal note]
from utils.losses import CompositeLoss  # [internal]
from utils.metrics import mae_rmse_mm, split_vec, chamfer_distance_mm
from utils.export import save_predictions
from utils.discovery import discover_patients
from utils.pid import normalize_pid
from utils.pad import pad_3d_list_cpu, pad_2d_list_cpu
from utils.speed import enable_speed_flags
from utils.norm import NormStats
from utils.vis import write_csv, plot_curves

# --------------------- Timestamped logging (stdout/stderr -> file) ---------------------
def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class _TeeStream:
    """
    [internal]
    tqdm [internal]
    """
    def __init__(self, stream, log_f):
        self.stream = stream
        self.log_f = log_f
        self._lock = threading.Lock()
        self._at_line_start = True

    def write(self, s: str):
        if s is None or s == "":
            return
        with self._lock:
            # [internal note]
            if s.startswith("\r"):
                self.stream.write(s)
                try:
                    self.log_f.write(s)
                except Exception:
                    pass
                self.stream.flush()
                try:
                    self.log_f.flush()
                except Exception:
                    pass
                return

            parts = s.split("\n")
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    # [internal note]
                    if self._at_line_start and part != "":
                        prefix = f"[{_ts()}] "
                        self.stream.write(prefix + part + "\n")
                        try:
                            self.log_f.write(prefix + part + "\n")
                        except Exception:
                            pass
                    else:
                        self.stream.write(part + "\n")
                        try:
                            self.log_f.write(part + "\n")
                        except Exception:
                            pass
                    self._at_line_start = True
                else:
                    # [internal note]
                    if part != "":
                        if self._at_line_start:
                            prefix = f"[{_ts()}] "
                            self.stream.write(prefix + part)
                            try:
                                self.log_f.write(prefix + part)
                            except Exception:
                                pass
                        else:
                            self.stream.write(part)
                            try:
                                self.log_f.write(part)
                            except Exception:
                                pass
                        self._at_line_start = False

            self.stream.flush()
            try:
                self.log_f.flush()
            except Exception:
                pass

    def flush(self):
        self.stream.flush()
        try:
            self.log_f.flush()
        except Exception:
            pass

def setup_logging(log_path="outputs/logs/train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # [internal note]
    try:
        f = open(log_path, "a", encoding="utf-8")
        active = log_path
    except PermissionError:
        alt = os.path.splitext(log_path)[0] + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        f = open(alt, "a", encoding="utf-8")
        active = alt

    sys.stdout = _TeeStream(sys.__stdout__, f)
    sys.stderr = _TeeStream(sys.__stderr__, f)
    print(f"Logging to: {active}")

# --------------------- CSV safe append (fix PermissionError) ---------------------
def _with_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"

def write_csv_safe(path: str, header: list, row: list, tag: str, retries: int = 3):
    """
    Windows [internal]
    [internal]
    [internal]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_err = None
    for k in range(retries):
        try:
            write_csv(path, header, row)
            return path
        except PermissionError as e:
            last_err = e
            time.sleep(0.4 * (k + 1))

    # fallback
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt = _with_suffix(path, f".{tag}.{ts}")
    try:
        write_csv(alt, header, row)
        print(f"[WARN] Cannot write '{path}' (likely opened/locked). "
              f"Switched to '{alt}' and continued.")
        return alt
    except Exception as e2:
        print(f"[ERROR] write_csv failed on both '{path}' and fallback '{alt}'.")
        raise e2 from last_err

# --------------------- Dataset (pickle-safe) ---------------------
import torch.utils.data as data
class PatientDataset(data.Dataset):
    def __init__(self, ids, sample_loader):
        self.ids = list(ids)
        self.sample_loader = sample_loader
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx): return self.sample_loader(self.ids[idx])

def setup_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def select_device(cfg):
    if cfg.get("device","auto") != "auto": return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flat_dim(S,N): return S*N*4 + S
def passthrough_collate(batch): return batch

# --------------------- compile fallback ---------------------
def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

def resolve_compile(model, cfg, who="model"):

    want_compile = bool(cfg.get("compile", True))
    backend = cfg.get("compile_backend", "auto")
    mode = cfg.get("compile_mode", "max-autotune")

    if not want_compile:
        print(f"[compile] {who}: disabled")
        return model, "eager"

    if backend == "eager":
        print(f"[compile] {who}: eager")
        return model, "eager"

    if backend == "auto":
        if _has_triton():
            try:
                model = torch.compile(model, mode=mode, backend="inductor")
                print(f"[compile] {who}: inductor")
                return model, "inductor"
            except Exception as e:
                print(f"[WARN] {who}: inductor failed: {e}; fallback aot_eager")
                try:
                    model = torch.compile(model, backend="aot_eager")
                    print(f"[compile] {who}: aot_eager")
                    return model, "aot_eager"
                except Exception as e2:
                    print(f"[WARN] {who}: aot_eager failed: {e2}; eager")
                    return model, "eager"
        else:
            print(f"[WARN] Triton not found -> {who} will use aot_eager instead of inductor.")
            try:
                model = torch.compile(model, backend="aot_eager")
                print(f"[compile] {who}: aot_eager")
                return model, "aot_eager"
            except Exception as e:
                print(f"[WARN] {who}: aot_eager failed: {e}; eager")
                return model, "eager"
    else:
        # [internal note]
        try:
            if backend in ("aot_eager", "aot_eager_decomp"):
                model = torch.compile(model, backend=backend)
            else:
                model = torch.compile(model, mode=mode, backend=backend)
            print(f"[compile] {who}: {backend}")
            return model, backend
        except Exception as e:
            print(f"[WARN] {who}: compile({backend}) failed: {e}; eager")
            return model, "eager"

# --------------------- dataloaders + demographics ---------------------
def build_dataloaders(cfg, ids):
    demo_path = cfg["demographics_path"]
    id_col = cfg.get("demographics_pid_column", "PID")

    df = pd.read_excel(demo_path) if demo_path.lower().endswith(".xlsx") else pd.read_csv(demo_path)
    df[id_col] = df[id_col].astype(str).map(normalize_pid)

    cols = cfg["demographics_features"]
    feat_df = df[cols].apply(pd.to_numeric, errors="coerce")
    col_means = feat_df.mean(numeric_only=True)
    feat_df = feat_df.fillna(col_means)

    demo_map = {row[id_col]: row[cols].to_numpy(dtype=np.float32)
                for _,row in pd.concat([df[[id_col]], feat_df], axis=1).iterrows()
                if isinstance(row[id_col], str) and len(row[id_col])>0}

    norm_by_orig = {orig: normalize_pid(orig) for orig in ids}
    keep_ids = [orig for orig, norm in norm_by_orig.items() if norm in demo_map]
    drop_ids = [orig for orig, norm in norm_by_orig.items() if norm not in demo_map]
    if drop_ids:
        os.makedirs("outputs/logs", exist_ok=True)
        with open("outputs/logs/missing_demographics.txt","w",encoding="utf-8") as f:
            f.write("Missing demographics (orig -> normalized):\n")
            for x in drop_ids: f.write(f"{x} -> {norm_by_orig[x]}\n")
        print(f"[WARN] {len(drop_ids)} patients skipped due to missing demographics. See outputs/logs/missing_demographics.txt")

    sample_loader = VesselSeqSampleLoader(cfg, demo_map)
    ntr = int(math.ceil(len(keep_ids)*float(cfg.get("train_ratio",0.8))))
    tr_ids, va_ids = keep_ids[:ntr], keep_ids[ntr:]

    pin = bool(cfg.get("pin_memory",True))
    nw  = int(cfg.get("num_workers", 8))
    bs  = int(cfg.get("batch_size", 16))
    pf  = int(cfg.get("prefetch_factor", 4))
    pw  = bool(cfg.get("persistent_workers", True) and nw>0)

    dl_args = dict(num_workers=nw, pin_memory=pin,
                   collate_fn=passthrough_collate,
                   persistent_workers=pw, prefetch_factor=pf)

    tr = DataLoader(PatientDataset(tr_ids, sample_loader), batch_size=bs, shuffle=True, **dl_args)
    va = DataLoader(PatientDataset(va_ids, sample_loader), batch_size=bs, shuffle=False, **dl_args)
    return tr, va, tr_ids, va_ids

# --------------------- batch to device ---------------------
def collate_to_device(batch, device, ch_last2d=True, ch_last3d=True):
    pids  = [b["pid"]   for b in batch]
    cts   = [b["ct"]    for b in batch]
    img2s = [b["img2d"] for b in batch]
    demos = [b["demo"]  for b in batch]
    tgts  = [b["target"]for b in batch]

    ct_cpu   = pad_3d_list_cpu(cts,  pin=True, dtype=torch.float32)
    img_cpu  = pad_2d_list_cpu(img2s, pin=True, dtype=torch.float32)

    nb = (device.type == "cuda")
    ct    = ct_cpu.to(device, non_blocking=nb)
    img2d = img_cpu.to(device, non_blocking=nb)

    if device.type == "cuda":
        if ch_last3d: ct = ct.contiguous(memory_format=torch.channels_last_3d)
        if ch_last2d: img2d = img2d.contiguous(memory_format=torch.channels_last)

    demo = torch.from_numpy(np.stack(demos,0).astype(np.float32, copy=False)).to(device=device, non_blocking=nb)
    y    = torch.from_numpy(np.stack(tgts, 0).astype(np.float32, copy=False)).to(device=device, non_blocking=nb)
    extra = {}
    return pids, ct, img2d, demo, y, extra

# [internal note]
def build_opt_sched(model, lr, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
    monitor = cfg.get("monitor_metric", "val_chamfer_mm")
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 5)), min_lr=float(cfg.get("min_lr", 1e-6))
    )
    return opt, sched, monitor

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = int(patience); self.min_delta = float(min_delta)
        self.mode = mode; self.best = None; self.count = 0
    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value; self.count = 0; return False
        improve = (value < self.best - self.min_delta) if self.mode=="min" else (value > self.best + self.min_delta)
        if improve: self.best = value; self.count = 0; return False
        self.count += 1
        return self.count > self.patience

# --------------------- Teacher ---------------------
def train_teacher(cfg, device, ids):
    S,N = int(cfg["num_segments"]), int(cfg["seq_points"])
    out_dim = flat_dim(S,N)
    tr, va, _, _ = build_dataloaders(cfg, ids)

    norm = NormStats(S, N,
                     mode=cfg.get("target_normalization","fixed"),
                     scale_coord_mm=float(cfg.get("scale_coord_mm",200.0)),
                     scale_diam_mm=float(cfg.get("scale_diam_mm",20.0)),
                     scale_len_mm=float(cfg.get("scale_len_mm",200.0)))

    enable_speed_flags()
    model = TeacherModel(demo_dim=len(cfg["demographics_features"]), out_dim=out_dim).to(device)
    try: model = model.to(memory_format=torch.channels_last_3d)
    except Exception: pass
    model, _ = resolve_compile(model, cfg, who="teacher")

    comp_loss = CompositeLoss(
        S,N,
        coord_w=float(cfg.get("coord_w",1.0)),
        diam_w=float(cfg.get("diam_w",1.0)),
        len_w=float(cfg.get("len_w",1.0)),
        use_huber=bool(cfg.get("use_huber_loss",False)),
        huber_delta=1.0,
        use_uncertainty=bool(cfg.get("uncertainty_weight",False)),
        diam_smooth_w1=float(cfg.get("diam_smooth_w1",0.0)),
        diam_smooth_w2=float(cfg.get("diam_smooth_w2",0.0)),
    ).to(device)

    opt, sched, monitor = build_opt_sched(model, float(cfg.get("lr_teacher",1e-4)), cfg)
    use_amp = bool(cfg.get("amp", True)) and device.type=="cuda"
    amp_dtype = torch.float16 if str(cfg.get("amp_dtype","fp16")).lower()!="bf16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    clip_norm  = float(cfg.get("clip_grad_norm", 1.0))
    early = EarlyStopper(int(cfg.get("earlystop_patience",10)), float(cfg.get("earlystop_delta",0.0)), "min")

    hist = {k:[] for k in ["epoch","train_mse","val_mse",
                           "val_coord_rmse_mm","val_diam_rmse_mm","val_len_rmse_mm",
                           "val_coord_mae_mm","val_diam_mae_mm","val_len_mae_mm",
                           "val_chamfer_mm"]}
    best_val = float("inf")
    chamfer_w = float(cfg.get("chamfer_w", 0.0))

    hist_path = str(cfg.get("history_teacher_csv", "outputs/history_teacher.csv"))

    for ep in range(1, int(cfg.get("epochs_teacher",100))+1):
        ep_t0 = time.time()
        print(f"[Teacher] epoch {ep} started.")

        model.train(); tot=0.0; seen=0
        opt.zero_grad(set_to_none=True)

        for i, raw in enumerate(tqdm(tr, desc=f"Teacher Train {ep}"), start=1):
            pids, ct, _, demo, y_mm, _ = collate_to_device(raw, device,
                                                           ch_last2d=bool(cfg.get("channels_last",True)),
                                                           ch_last3d=bool(cfg.get("channels_last_3d",True)))
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_mm = model(ct, demo)     # [internal]
                pred = norm.apply_t(pred_mm)  # [internal]
                y    = norm.apply_t(y_mm)

                extra_terms = []
                if chamfer_w > 0.0:
                    pc, _, _ = split_vec(pred_mm, S,N)
                    yc, _, _ = split_vec(y_mm,   S,N)
                    extra_terms.append(chamfer_w * chamfer_distance_mm(pc, yc))  # [internal]

                loss_total, _ = comp_loss(pred, y, extra_terms=extra_terms)
                loss = loss_total / grad_accum

            if loss.ndim != 0:
                loss = loss.mean()

            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            if i % grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            tot += float(loss.item())*grad_accum*ct.size(0); seen += ct.size(0)

        tr_mse = tot/max(1,seen)

        # [internal note]
        model.eval(); val=0.0; m=0
        rm = {"coord_rmse_mm":0.0,"diam_rmse_mm":0.0,"len_rmse_mm":0.0,
              "coord_mae_mm":0.0, "diam_mae_mm":0.0, "len_mae_mm":0.0,
              "chamfer_mm":0.0}
        c=0
        with torch.no_grad():
            for raw in tqdm(va, desc=f"Teacher Val {ep}"):
                pids, ct, _, demo, y_mm, _ = collate_to_device(raw, device,
                                                               ch_last2d=bool(cfg.get("channels_last",True)),
                                                               ch_last3d=bool(cfg.get("channels_last_3d",True)))
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred_mm = model(ct, demo)
                    pred = norm.apply_t(pred_mm); y = norm.apply_t(y_mm)
                    sup,_ = CompositeLoss(S,N).core(pred, y)  # [internal]

                if sup.ndim != 0: sup = sup.mean()
                val += float(sup.item())*ct.size(0); m += ct.size(0)

                stats = mae_rmse_mm(pred_mm, y_mm, S,N)
                rm["coord_rmse_mm"] += stats["coord_rmse_mm"]
                rm["diam_rmse_mm"]  += stats["diam_rmse_mm"]
                rm["len_rmse_mm"]   += stats["len_rmse_mm"]
                rm["coord_mae_mm"]  += stats["coord_mae_mm"]
                rm["diam_mae_mm"]   += stats["diam_mae_mm"]
                rm["len_mae_mm"]    += stats["len_mae_mm"]
                if chamfer_w > 0.0:
                    pc, _, _ = split_vec(pred_mm, S,N)
                    yc, _, _ = split_vec(y_mm,   S,N)
                    rm["chamfer_mm"] += float(chamfer_distance_mm(pc, yc).item())
                c += 1

        va_mse = val/max(1,m)
        for k in rm: rm[k] /= max(1,c)

        # [internal note]
        hist["epoch"].append(ep); hist["train_mse"].append(tr_mse); hist["val_mse"].append(va_mse)
        hist["val_coord_rmse_mm"].append(rm["coord_rmse_mm"])
        hist["val_diam_rmse_mm"].append(rm["diam_rmse_mm"])
        hist["val_len_rmse_mm"].append(rm["len_rmse_mm"])
        hist["val_coord_mae_mm"].append(rm["coord_mae_mm"])
        hist["val_diam_mae_mm"].append(rm["diam_mae_mm"])
        hist["val_len_mae_mm"].append(rm["len_mae_mm"])
        hist["val_chamfer_mm"].append(rm["chamfer_mm"])

        hist_path = write_csv_safe(
            hist_path,
            ["epoch","train_mse","val_mse",
             "val_coord_rmse_mm","val_diam_rmse_mm","val_len_rmse_mm",
             "val_coord_mae_mm","val_diam_mae_mm","val_len_mae_mm","val_chamfer_mm"],
            [ep, tr_mse, va_mse,
             rm["coord_rmse_mm"], rm["diam_rmse_mm"], rm["len_rmse_mm"],
             rm["coord_mae_mm"],  rm["diam_mae_mm"],  rm["len_mae_mm"], rm["chamfer_mm"]],
            tag="teacher"
        )
        plot_curves(hist, "outputs/figs/teacher_curves.png", "Teacher metrics")

        # [internal note]
        monitor_map = {
            "val_mse": va_mse,
            "val_coord_rmse_mm": rm["coord_rmse_mm"],
            "val_diam_rmse_mm":  rm["diam_rmse_mm"],
            "val_len_rmse_mm":   rm["len_rmse_mm"],
            "val_chamfer_mm":    rm["chamfer_mm"],
        }
        monitor_value = monitor_map.get(cfg.get("monitor_metric","val_chamfer_mm"), va_mse)
        sched.step(monitor_value)

        if monitor_value < best_val:
            best_val = monitor_value
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), cfg["teacher_ckpt"])

        ep_min = (time.time() - ep_t0) / 60.0
        print(f"[Teacher] {ep}: MSE(norm) train {tr_mse:.4f}  val {va_mse:.4f}  |  "
              f"RMSE(mm) coord {rm['coord_rmse_mm']:.2f}  diam {rm['diam_rmse_mm']:.2f}  len {rm['len_rmse_mm']:.2f}  |  "
              f"Chamfer(mm) {rm['chamfer_mm']:.2f}  |  time {ep_min:.2f} min")

        if early.step(monitor_value):
            print(f"[EarlyStop] teacher stopped at epoch {ep} (best {best_val:.4f})")
            break

    return model

# --------------------- Student (with distillation) ---------------------
def train_student(cfg, device, ids, teacher=None):
    S,N = int(cfg["num_segments"]), int(cfg["seq_points"])
    out_dim = flat_dim(S,N)
    C = len(cfg.get("projection_channels", ["bin","thickness_mm","depth_mm","edt_mm"])) * 2
    tr, va, _, _ = build_dataloaders(cfg, ids)

    norm = NormStats(S, N,
                     mode=cfg.get("target_normalization","fixed"),
                     scale_coord_mm=float(cfg.get("scale_coord_mm",200.0)),
                     scale_diam_mm=float(cfg.get("scale_diam_mm",20.0)),
                     scale_len_mm=float(cfg.get("scale_len_mm",200.0)))

    enable_speed_flags()
    stu = StudentModel(demo_dim=len(cfg["demographics_features"]), out_dim=out_dim, in_channels=C).to(device)
    try: stu = stu.to(memory_format=torch.channels_last)
    except Exception: pass
    stu, _ = resolve_compile(stu, cfg, who="student")

    if teacher is None:
        teacher = TeacherModel(demo_dim=len(cfg["demographics_features"]), out_dim=out_dim).to(device)
        ckpt = torch.load(cfg["teacher_ckpt"], map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        clean = {}
        for k,v in (ckpt.items() if isinstance(ckpt, dict) else []):
            if k.startswith("_orig_mod."): clean[k[10:]] = v
            elif k.startswith("model."):   clean[k[6:]]  = v
            else:                          clean[k]      = v
        teacher.load_state_dict(clean, strict=False)
    teacher.eval()

    comp_loss = CompositeLoss(
        S,N,
        coord_w=float(cfg.get("coord_w",1.0)),
        diam_w=float(cfg.get("diam_w",1.0)),
        len_w=float(cfg.get("len_w",1.0)),
        use_huber=bool(cfg.get("use_huber_loss",False)),
        huber_delta=1.0,
        use_uncertainty=False,  # [internal]
        diam_smooth_w1=float(cfg.get("diam_smooth_w1",0.0)),
        diam_smooth_w2=float(cfg.get("diam_smooth_w2",0.0)),
    ).to(device)

    alpha_all  = float(cfg.get("distill_weight_all", 0.3))
    alpha_diam = float(cfg.get("distill_weight_diam",0.8))
    mse = torch.nn.MSELoss(reduction="mean")

    opt, sched, monitor = build_opt_sched(stu, float(cfg.get("lr_student",1e-4)), cfg)
    use_amp = bool(cfg.get("amp", True)) and device.type=="cuda"
    amp_dtype = torch.float16 if str(cfg.get("amp_dtype","fp16")).lower()!="bf16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    clip_norm  = float(cfg.get("clip_grad_norm", 1.0))
    early = EarlyStopper(int(cfg.get("earlystop_patience",10)), float(cfg.get("earlystop_delta",0.0)), "min")

    hist = {k:[] for k in ["epoch","train_mse","val_mse",
                           "val_coord_rmse_mm","val_diam_rmse_mm","val_len_rmse_mm",
                           "val_coord_mae_mm","val_diam_mae_mm","val_len_mae_mm",
                           "val_chamfer_mm"]}
    best_val = float("inf")
    chamfer_w = float(cfg.get("chamfer_w", 0.0))

    hist_path = str(cfg.get("history_student_csv", "outputs/history_student.csv"))

    for ep in range(1, int(cfg.get("epochs_student",100))+1):
        ep_t0 = time.time()
        print(f"[Student] epoch {ep} started.")

        stu.train(); tot=0.0; seen=0
        opt.zero_grad(set_to_none=True)

        for i, raw in enumerate(tqdm(tr, desc=f"Student Train {ep}"), start=1):
            pids, ct, img2d, demo, y_mm, _ = collate_to_device(raw, device,
                                                               ch_last2d=bool(cfg.get("channels_last",True)),
                                                               ch_last3d=bool(cfg.get("channels_last_3d",True)))
            with torch.no_grad():
                t_mm = teacher(ct, demo)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                s_mm = stu(img2d, demo)
                s_n = norm.apply_t(s_mm); y_n = norm.apply_t(y_mm); t_n = norm.apply_t(t_mm)

                extra_terms = []
                if chamfer_w > 0.0:
                    sc, _, _ = split_vec(s_mm, S,N)
                    yc, _, _ = split_vec(y_mm, S,N)
                    extra_terms.append(chamfer_w * chamfer_distance_mm(sc, yc))

                sup,_ = comp_loss(s_n, y_n, extra_terms=extra_terms)  # [internal]
                # [internal note]
                s_cd, s_d, _ = split_vec(s_n, S,N)
                t_cd, t_d, _ = split_vec(t_n, S,N)
                distill = alpha_all  * mse(s_n, t_n) + alpha_diam * mse(s_d, t_d)

                loss = (sup + distill) / grad_accum

            if loss.ndim != 0:
                loss = loss.mean()

            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(stu.parameters(), max_norm=clip_norm)
            if i % grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            tot += float(loss.item())*grad_accum*img2d.size(0); seen += img2d.size(0)

        tr_mse = tot/max(1,seen)

        # [internal note]
        stu.eval(); val=0.0; m=0
        rm = {"coord_rmse_mm":0.0,"diam_rmse_mm":0.0,"len_rmse_mm":0.0,
              "coord_mae_mm":0.0, "diam_mae_mm":0.0, "len_mae_mm":0.0,
              "chamfer_mm":0.0}
        c=0
        with torch.no_grad():
            for raw in tqdm(va, desc=f"Student Val {ep}"):
                pids, ct, img2d, demo, y_mm, _ = collate_to_device(raw, device,
                                                                   ch_last2d=bool(cfg.get("channels_last",True)),
                                                                   ch_last3d=bool(cfg.get("channels_last_3d",True)))
                s_mm = stu(img2d, demo)
                s_n  = norm.apply_t(s_mm); y_n = norm.apply_t(y_mm)
                sup,_ = CompositeLoss(S,N).core(s_n, y_n)

                if sup.ndim != 0: sup = sup.mean()
                val += float(sup.item())*img2d.size(0); m += img2d.size(0)

                stats = mae_rmse_mm(s_mm, y_mm, S,N)
                rm["coord_rmse_mm"] += stats["coord_rmse_mm"]
                rm["diam_rmse_mm"]  += stats["diam_rmse_mm"]
                rm["len_rmse_mm"]   += stats["len_rmse_mm"]
                rm["coord_mae_mm"]  += stats["coord_mae_mm"]
                rm["diam_mae_mm"]   += stats["diam_mae_mm"]
                rm["len_mae_mm"]    += stats["len_mae_mm"]
                if chamfer_w > 0.0:
                    sc, _, _ = split_vec(s_mm, S,N)
                    yc, _, _ = split_vec(y_mm, S,N)
                    rm["chamfer_mm"] += float(chamfer_distance_mm(sc, yc).item())
                c += 1

        va_mse = val/max(1,m)
        for k in rm: rm[k] /= max(1,c)

        # [internal note]
        hist["epoch"].append(ep); hist["train_mse"].append(tr_mse); hist["val_mse"].append(va_mse)
        hist["val_coord_rmse_mm"].append(rm["coord_rmse_mm"])
        hist["val_diam_rmse_mm"].append(rm["diam_rmse_mm"])
        hist["val_len_rmse_mm"].append(rm["len_rmse_mm"])
        hist["val_coord_mae_mm"].append(rm["coord_mae_mm"])
        hist["val_diam_mae_mm"].append(rm["diam_mae_mm"])
        hist["val_len_mae_mm"].append(rm["len_mae_mm"])
        hist["val_chamfer_mm"].append(rm["chamfer_mm"])

        hist_path = write_csv_safe(
            hist_path,
            ["epoch","train_mse","val_mse",
             "val_coord_rmse_mm","val_diam_rmse_mm","val_len_rmse_mm",
             "val_coord_mae_mm","val_diam_mae_mm","val_len_mae_mm","val_chamfer_mm"],
            [ep, tr_mse, va_mse,
             rm["coord_rmse_mm"], rm["diam_rmse_mm"], rm["len_rmse_mm"],
             rm["coord_mae_mm"],  rm["diam_mae_mm"],  rm["len_mae_mm"], rm["chamfer_mm"]],
            tag="student"
        )
        plot_curves(hist, "outputs/figs/student_curves.png", "Student metrics")

        monitor_map = {
            "val_mse": va_mse,
            "val_coord_rmse_mm": rm["coord_rmse_mm"],
            "val_diam_rmse_mm":  rm["diam_rmse_mm"],
            "val_len_rmse_mm":   rm["len_rmse_mm"],
            "val_chamfer_mm":    rm["chamfer_mm"],
        }
        monitor_value = monitor_map.get(cfg.get("monitor_metric","val_chamfer_mm"), va_mse)
        sched.step(monitor_value)
        if monitor_value < best_val:
            best_val = monitor_value
            os.makedirs("outputs", exist_ok=True)
            torch.save(stu.state_dict(), cfg["student_ckpt"])

        ep_min = (time.time() - ep_t0) / 60.0
        print(f"[Student] {ep}: MSE(norm) train {tr_mse:.4f}  val {va_mse:.4f}  |  "
              f"RMSE(mm) coord {rm['coord_rmse_mm']:.2f}  diam {rm['diam_rmse_mm']:.2f}  len {rm['len_rmse_mm']:.2f}  |  "
              f"Chamfer(mm) {rm['chamfer_mm']:.2f}  |  time {ep_min:.2f} min")

        if early.step(monitor_value):
            print(f"[EarlyStop] student stopped at epoch {ep} (best {best_val:.4f})")
            break

    return stu

# [internal note]
def ckpt_load_flexible(model, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    clean = {}
    for k,v in (sd.items() if isinstance(sd, dict) else []):
        if k.startswith("_orig_mod."): clean[k[10:]] = v
        elif k.startswith("model."):   clean[k[6:]]  = v
        else:                          clean[k]      = v
    msg = model.load_state_dict(clean, strict=False)
    print(f"[CKPT] loaded '{path}'. matched={len(clean)}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    return model

def export_predictions(cfg, device, ids):
    S,N = int(cfg["num_segments"]), int(cfg["seq_points"])
    out_dim = flat_dim(S,N)
    C = len(cfg.get("projection_channels", ["bin","thickness_mm","depth_mm","edt_mm"])) * 2
    enable_speed_flags()
    stu = StudentModel(demo_dim=len(cfg["demographics_features"]), out_dim=out_dim, in_channels=C).to(device)
    try: stu = stu.to(memory_format=torch.channels_last)
    except Exception: pass
    stu = ckpt_load_flexible(stu, cfg["student_ckpt"], device)
    stu.eval()

    tr, va, tr_ids, va_ids = build_dataloaders(cfg, ids)

    def _run(dloader, tag):
        with torch.no_grad():
            for raw in tqdm(dloader, desc=f"Export predictions ({tag})"):
                pids, ct, img2d, demo, _ , _ = collate_to_device(raw, device,
                                                                 ch_last2d=bool(cfg.get("channels_last",True)),
                                                                 ch_last3d=bool(cfg.get("channels_last_3d",True)))
                out_mm = stu(img2d, demo)
                for i,pid in enumerate(pids):
                    vec = out_mm[i].detach().cpu().numpy()
                    save_predictions(pid, vec, out_dir="outputs", S=S, N=N, segment_names=cfg["segment_names"])
    _run(tr, "train")
    _run(va, "val")

# --------------------- main ---------------------
if __name__ == "__main__":
    setup_logging("outputs/logs/train.log")

    cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    setup_seed(int(cfg.get("seed",42)))
    device = select_device(cfg)
    print("Using device:", device)

    ids = discover_patients(
        cfg["data_root"],
        cfg["origin_subdir"],
        cfg["models_subdir"],
        cfg["tables_subdir"],
        cfg.get("ct_filename", None),
        cfg.get("surface_seg_filename", None),
        cfg["segment_names"],
        report_path="outputs/logs/discovery_report.txt"
    )
    if not ids:
        raise SystemExit("No valid patients discovered under data_root. "
                         "[see source]")
    random.Random(cfg.get('seed',42)).shuffle(ids)
    print(f"Discovered {len(ids)} patients.")

    mode = str(cfg.get("mode","train_both")).lower()
    if mode == "train_teacher":
        train_teacher(cfg, device, ids)
    elif mode == "train_student":
        if not os.path.exists(cfg["teacher_ckpt"]):
            raise SystemExit("Teacher checkpoint not found.")
        train_student(cfg, device, ids)
    elif mode == "train_both":
        t = train_teacher(cfg, device, ids)
        train_student(cfg, device, ids, teacher=t)
    elif mode == "eval":
        export_predictions(cfg, device, ids)
    else:
        raise SystemExit("Unknown mode: "+mode)
