"""python tools/paired.py "heroA" "heroB" field1,field2 tables"""
import sys, statistics as S
sys.path.insert(0, ".")
from concurrent.futures import ProcessPoolExecutor
from bench import run_points
if __name__ == "__main__":
    a, b, fields, n = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
    sd = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    with ProcessPoolExecutor(16) as ex:
        for f in fields:
            pa = run_points(a, f, n, seed=sd, pool=ex); pb = run_points(b, f, n, seed=sd, pool=ex)
            d = [y - x for x, y in zip(pa, pb)]
            print(f"{f:<11} A {S.mean(pa):.3f}  B {S.mean(pb):.3f}  B-A {S.mean(d):+.3f} ± {S.stdev(d)/len(d)**.5:.3f}", flush=True)
