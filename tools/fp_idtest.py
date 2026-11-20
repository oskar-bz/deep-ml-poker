"""Identification test: signatures from SIG_SOURCE, tested on TEST files.
python tools/fp_idtest.py SIG_SOURCE TEST_GLOB   (SIG_SOURCE = 'current' -> bots/fingerprints.py, or a folder)"""
import glob, sys
sys.path.insert(0, ".")
import bots.fp_core as C
from tools.build_profiles import build_signatures, identification_test, FINALISTS
src, test = sys.argv[1], sys.argv[2]
if src == "current":
    from bots.fingerprints import FP_SIGNATURES as sig
else:
    sig = build_signatures(sorted(glob.glob(src + "/*.json")))
files = sorted(glob.glob(test))
for label, ms in (("old rule (min sized 0)", 0), ("new rule (min sized 2)", 2)):
    C.FP_MIN_SIZED = ms
    res, wrong, never = identification_test(sig, files)
    print(label, "games", len(files))
    for b in FINALISTS:
        L = sorted(res[b])
        med = L[len(L)//2] if L else -1
        print(f"  {b:<16} ok {len(L):4d} median {med:3d}  wrong {sum(v for (t, w), v in wrong.items() if t == b):3d}  never {never[b]:3d}")
    if wrong: print("  confusions:", wrong.most_common(6))
