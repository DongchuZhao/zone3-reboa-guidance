# utils/speed.py
import torch

def enable_speed_flags():
    # [internal note]
    torch.backends.cudnn.benchmark = True
    # [internal note]
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    # [internal note]
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
