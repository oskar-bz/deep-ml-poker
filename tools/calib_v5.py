"""Per-bot points in the v5 field (hero = built file) vs real round-4 points.  python tools/calib_v5.py BUILD [tables]"""
import sys, random, importlib.util
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
REAL = {"HERO": 3.08, "mw": 2.86, "Daksh Malhan": 2.48, "Vaibhaav Tiwari": 1.84, "Henrik Gombos": 1.76,
        "Lorenzo P.": 1.75, "Kashyap Patel": 1.68, "Shresth Rana": 1.55}


def run(args):
    path, seeds = args
    from poker_engine.simulator import SitAndGoTable
    from pokerbot_sdk import SDKBotDriver
    from bots.sdk_field import R5ModelBot
    spec = importlib.util.spec_from_file_location("h", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res = defaultdict(list)
    for sd in seeds:
        rng = random.Random(sd)
        names = ["HERO"] + [n for n in REAL if n != "HERO"]
        bots = [SDKBotDriver(m.MyBot(), "HERO")] + [SDKBotDriver(R5ModelBot(n, rng.randrange(1 << 30)), n) for n in names[1:]]
        order = list(range(8)); rng.shuffle(order)
        r = SitAndGoTable([bots[i] for i in order], seed=sd).play_tournament()
        for k, i in enumerate(order):
            res[names[i]].append(r.points[k])
    return res


if __name__ == "__main__":
    path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    allr = defaultdict(list)
    with ProcessPoolExecutor(16) as ex:
        for r in ex.map(run, [(path, list(range(9000 + k, 9000 + n, 16))) for k in range(16)]):
            for k, v in r.items(): allr[k] += v
    xs, ys = [], []
    for k in REAL:
        v = allr[k]; m = sum(v) / len(v); xs.append(m); ys.append(REAL[k])
        print(f"{k:<16} sim {m:5.2f}   real {REAL[k]:5.2f}")
    mx, my = sum(xs) / 8, sum(ys) / 8
    c = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** .5
    print("corr", round(c, 3))
