import torch, sys, os

def strip(path_in, path_out=None):
    obj = torch.load(path_in, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        sd = obj["model"]
        payload = obj
    else:
        sd = obj
        payload = {"model": sd}

    def _strip(sd):
        new = {}
        for k,v in sd.items():
            if k.startswith("_orig_mod."): k = k[len("_orig_mod."):]
            if k.startswith("module."):    k = k[len("module."):]
            if k.startswith("model."):     k = k[len("model."):]
            new[k] = v
        return new

    payload["model"] = _strip(sd)
    out = path_out or (os.path.splitext(path_in)[0] + "_clean.pth")
    torch.save(payload, out)
    print(f"wrote: {out}")

if __name__ == "__main__":
    strip(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
