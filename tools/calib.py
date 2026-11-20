"""Calibrate model opponents against round-1 reality.
python tools/calib.py hero|pure [tables] [hero_file]
hero: seat 0 = our bot file (default contest/round1_submission_bot.py), 7 random R2 models.
pure: 8 random R2 models. Prints avg points per bot name vs real round-1 points."""
import sys, json, glob, random, importlib.util
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")


def _load(path):
    spec = importlib.util.spec_from_file_location("hero_mod_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run(args):
    mode, seeds, hero_file, fname = args
    from poker_engine.simulator import SitAndGoTable
    from pokerbot_sdk import SDKBotDriver
    from bots.sdk_field import R2ModelBot
    mod = _load(hero_file) if mode == "hero" else None
    res = defaultdict(list)
    for sd in seeds:
        rng = random.Random(sd)
        bots, names = [], []
        if mode == "hero":
            bots.append(SDKBotDriver(mod.MyBot(), "hero")); names.append("HERO")
        while len(bots) < 8:
            b = R2ModelBot(None, rng.randrange(1 << 30), fname)
            bots.append(SDKBotDriver(b, b.name)); names.append(b.name)
        order = list(range(8)); rng.shuffle(order)
        bots = [bots[i] for i in order]; names = [names[i] for i in order]
        out = SitAndGoTable(bots, seed=sd).play_tournament()
        pts = out.points
        for i, n in enumerate(names):
            res[n].append(pts[i])
    return dict(res)


REAL_DIR = "data/r1"


def real_points():
    tot = defaultdict(lambda: [0.0, 0])
    for f in glob.glob(REAL_DIR + "/*.json"):
        g = json.load(open(f, encoding="utf-8"))["game"]
        for s, p in enumerate(g["players"]):
            n = p["name"].replace(" (copy)", "")
            tot[n][0] += g["points"][s]; tot[n][1] += 1
    return {n: v[0] / v[1] for n, v in tot.items()}


if __name__ == "__main__":
    mode = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    hero_file = sys.argv[3] if len(sys.argv) > 3 else "contest/round1_submission_bot.py"
    fname = sys.argv[4] if len(sys.argv) > 4 else "r1_models2.json"
    globals()["REAL_DIR"] = sys.argv[5] if len(sys.argv) > 5 else "data/r1"
    W = 16
    seeds = list(range(7000, 7000 + n))
    agg = defaultdict(list)
    with ProcessPoolExecutor(W) as ex:
        for r in ex.map(run, [(mode, seeds[i::W], hero_file, fname) for i in range(W)]):
            for k, v in r.items():
                agg[k].extend(v)
    real = real_points(); real["HERO"] = real.get("Oskar Behzadpour")
    rows = sorted(agg.items(), key=lambda kv: -real.get(kv[0], 0))
    import statistics as S
    xs, ys = [], []
    print(f"{'bot':<20}{'real':>6}{'model':>7}{'n':>6}")
    for k, v in rows:
        if len(v) < 30: continue
        print(f"{k[:19]:<20}{real.get(k, 0):6.2f}{S.mean(v):7.2f}{len(v):6d}")
        if k != "HERO": xs.append(real.get(k, 0)); ys.append(S.mean(v))
    mx, my = S.mean(xs), S.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    print("corr(real, model) over models: %.2f" % (cov / (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5))
