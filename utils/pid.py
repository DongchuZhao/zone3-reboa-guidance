# [internal note]

import re

_pat_asct = re.compile(r'(ASCT)\s*(\d+)', re.IGNORECASE)
_pat_sct  = re.compile(r'(SCT)\s*(\d+)', re.IGNORECASE)
_pat_s    = re.compile(r'(^|[^A-Z0-9])(S)\s*(\d+)', re.IGNORECASE)  # [internal]

def normalize_pid(text: str) -> str:
    """
    [internal]
    [internal]
      "asct 00123" -> "ASCT00123"
      "sct123"     -> "SCT123"
      " s 45 "     -> "S45"
      "patientX"   -> "PATIENTX"  ([internal]
    """
    if text is None:
        return ""
    s = str(text).strip().upper()

    # [internal note]
    m = _pat_asct.search(s)
    if m:
        return f"ASCT{m.group(2)}"

    m = _pat_sct.search(s)
    if m:
        return f"SCT{m.group(2)}"

    # [internal note]
    m = _pat_s.search(s)
    if m:
        return f"S{m.group(3)}"

    # [internal note]
    return s

