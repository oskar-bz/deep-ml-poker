"""Compare situation-key distribution + 3-bet rate for mw facing one open, deep: real (r2) vs sim."""
import glob, json, random, sys
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from tools.fit_r1_models2 import pre_decile, depth, pos_bucket
WHO = sys.argv[1] if len(sys.argv) > 1 else "mw"
real = defaultdict(lambda: [0, 0])
for f in glob.glob("data/r2/*.json"):
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        bb = h["b"][1]
        dealt = [x for x in range(8) if h["s"][x] > 0]
        blinds = [a[1] for a in h["a"] if a[2] == "b"]
        bbs = blinds[1] if len(blinds) > 1 else blinds[0]
        nr, cur, jam1 = 0, bb, False; folded = set()
        for a in (x for x in h["a"] if x[0] == 0 and x[2] != "b"):
            s = a[1]
            if nm[s] == WHO and nr == 1 and depth(h["s"][s] / bb) == 3:
                nk = "j" if jam1 else ("1L" if cur > 2.6 * bb else "1")
                live = [x for x in dealt if x not in folded]
                behind = 0
                if s != bbs:
                    x = s
                    for _ in range(8):
                        x = (x + 1) % 8
                        if x in live and x != s: behind += 1
                        if x == bbs: break
                p = 0 if s == bbs else pos_bucket(behind)
                k = (nk, p); real[k][0] += 1; real[k][1] += a[2] in "RA"
            if a[2] == "F": folded.add(s)
            if a[2] in "RA" and a[3] > cur:
                if nr == 0: jam1 = a[2] == "A"
                nr += 1; cur = a[3]
import bots.sdk_field as F
from pokerbot_sdk import SDKBotDriver
from poker_engine.simulator import SitAndGoTable
import bench
sim = defaultdict(lambda: [0, 0])
orig = F.R2ModelBot._pick
def pick(self, p, frac=None):
    i = orig(self, p, frac)
    c = getattr(self, "_cur_key", None)
    if c is not None:
        sim[c][0] += 1; sim[c][1] += i >= 2
        self._cur_key = None
    return i
F.R2ModelBot._pick = pick
orig_get = dict.get
class KeyDict(dict):
    pass
orig_act = F.R2ModelBot.act
def act(self, state):
    if self.name == WHO and state.street == "preflop" and not isinstance(self.pre, KeyDict):
        kd = KeyDict(self.pre); bot = self
        def g(key, default=None, _kd=kd):
            if key[0] in "1j" and key[-2] == "3":
                nk = key[:-2]; bot._cur_key = (nk, int(key[-1]))
            return dict.get(_kd, key, default)
        kd.get = g
        self.pre = kd
    return orig_act(self, state)
F.R2ModelBot.act = act
reg = bench._registry()
for sd in range(40):
    random.seed(sd)
    bots = [reg["submission"]()] + [SDKBotDriver(F.R2ModelBot(WHO, random.randrange(1 << 30), "r2_models2.json"), "x") for _ in range(7)]
    SitAndGoTable(bots, seed=sd).play_tournament()
tr = sum(v[0] for v in real.values()); ts = sum(v[0] for v in sim.values())
print("key        real share  real 3b%   sim share  sim 3b%")
for k in sorted(set(real) | set(sim), key=str):
    r = real[k]; s_ = sim[k]
    print(f"{str(k):<12}{r[0]/tr:9.1%}{r[1]/max(1,r[0]):9.0%}   {s_[0]/max(1,ts):9.1%}{s_[1]/max(1,s_[0]):9.0%}")
