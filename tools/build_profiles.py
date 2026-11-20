"""Opponent fingerprints + profiles from real replays -> bots/fingerprints.py (embedded in the build).

Signatures (per bot, categorical, from public actions):
  OPEN : open raise-to in bb (non-all-in first raise), rounded to 0.05
  UNOP : first action in an unopened pot when not in the big blind: F / C (limp) / R / A
  X3B  : 3-bet raise-to / open raise-to, rounded to 0.1 (non-all-in)
  BET  : postflop first bet / pot before it, rounded to 0.02
Profiles (per bot): counters of OUR core, accumulated by feeding the real hands in platform format through
the round-4 build (so the numbers use exactly the definitions the exploits read).

python tools/build_profiles.py data/r3 [games]
"""
import glob
import json
import math
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from bots.fp_core import fp_round  # noqa: E402

FINALISTS = ("mw", "Daksh Malhan", "Vaibhaav Tiwari", "Lorenzo P.", "Henrik Gombos", "Shresth Rana", "Kashyap Patel")
ME = "Oskar Behzadpour"
PROFILE_KEYS = ("faced_steal", "folded_steal", "threebet_count", "faced_3bet", "folded_3bet", "fourbet_count",
                "faced_jam", "called_jam", "post_faced_bet", "post_folded_bet", "vu_faced", "vu_3b",
                "f4_n", "f4_fold", "check_faced_bet", "check_folded_bet", "faced_xr", "folded_xr", "hands_dealt", "vpip_count", "pfr_count")


def features_from_hand(h, seat_names=None):
    """Yield (seat, feature, value) from one replay hand (public information only)."""
    bb = h["b"][1]
    blinds = [a[1] for a in h["a"] if a[2] == "b"]
    bb_seat = blinds[1] if len(blinds) > 1 else (blinds[0] if blinds else -1)
    pre = [a for a in h["a"] if a[0] == 0 and a[2] != "b"]
    nr = 0; open_amt = 0
    for a in pre:
        s, c, amt = a[1], a[2], a[3]
        if nr == 0 and s != bb_seat:
            yield s, "UNOP", c if c in "FCRA" else "F"
            if c == "R":
                yield s, "OPEN", fp_round(amt / bb, 0.05, 2)
        if nr == 1 and c == "R" and open_amt > 0:
            yield s, "X3B", fp_round(amt / open_amt, 0.1, 1)
        if c in "RA":
            if nr == 0:
                open_amt = amt
            nr += 1
    prev = 0
    for st in range(4):
        com = defaultdict(int); bet = False
        for a in (x for x in h["a"] if x[0] == st):
            if st > 0 and not bet and a[2] == "B":
                yield a[1], "BET", fp_round(a[3] / max(1, prev), 0.02, 2)
            if a[2] in "BRA":
                bet = True
            if a[2] in "CBRAb" and a[3]:
                com[a[1]] = max(com[a[1]], a[3])
        prev += sum(com.values())


def build_signatures(files):
    cnt = {b: defaultdict(Counter) for b in FINALISTS}
    for f in files:
        g = json.load(open(f, encoding="utf-8"))["game"]
        nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
        for h in g["hands"]:
            for s, feat, val in features_from_hand(h):
                if nm[s] in cnt:
                    cnt[nm[s]][feat][val] += 1
    sig = {}
    for b, feats in cnt.items():
        sig[b] = {}
        for feat, c in feats.items():
            tot = sum(c.values())
            keep = {str(v): n / tot for v, n in c.items() if n / tot >= 0.004}
            sig[b][feat] = {k: round(v, 4) for k, v in keep.items()}
    return sig


def build_profiles(files, build_path):
    import importlib.util
    from tools import replay_feed as RF
    spec = importlib.util.spec_from_file_location("built_prof", build_path)
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    RF.SHOWN_ONLY = True
    agg = {b: Counter() for b in FINALISTS}
    errs = []
    for f in files:
        g = json.load(open(f, encoding="utf-8"))["game"]
        for seat in [i for i, p in enumerate(g["players"]) if p["name"] == ME]:
            core, names = RF.feed_game(S, g, seat, True, errs)
            for s in range(8):
                if s != seat and names[s] in agg:
                    for k in PROFILE_KEYS:
                        v = getattr(core, k)[s]
                        agg[names[s]][k] += v
    return agg, errs


def identification_test(sig, files):
    """Replay public actions seat by seat and report hands needed to identify each finalist."""
    import importlib
    FP = importlib.import_module("bots.fingerprints") if False else None
    from bots.fp_core import Fingerprinter
    res = defaultdict(list); wrong = Counter(); never = Counter()
    for f in files:
        g = json.load(open(f, encoding="utf-8"))["game"]
        nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
        fp = Fingerprinter(sig)
        done = {}
        for hi, h in enumerate(g["hands"]):
            for s, feat, val in features_from_hand(h):
                fp.observe(s, feat, val)
            for s in range(8):
                if s in done:
                    continue
                who = fp.identify(s)
                if who is not None:
                    done[s] = (hi, who)
        for s in range(8):
            if nm[s] not in FINALISTS:
                continue
            if s in done:
                hi, who = done[s]
                if who == nm[s]:
                    res[nm[s]].append(hi)
                else:
                    wrong[(nm[s], who)] += 1
            else:
                never[nm[s]] += 1
    return res, wrong, never


if __name__ == "__main__":
    folder = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    files = sorted(glob.glob(folder + "/*.json"))
    train, test = files[: n // 2], files[n // 2: n]
    sig = build_signatures(train)
    json.dump(sig, open("data/fp_signatures.json", "w"), indent=0)
    res, wrong, never = identification_test(sig, test)
    print("identification on held-out games (hand index when identified):")
    for b in FINALISTS:
        L = sorted(res[b])
        if L:
            print(f"  {b:<16} identified {len(L):4d}  median hand {L[len(L)//2]:3d}  p90 {L[int(.9*(len(L)-1))]:3d}"
                  f"   wrong {sum(v for (t, w), v in wrong.items() if t == b):3d}  never {never[b]:3d}")
        else:
            print(f"  {b:<16} never identified ({never[b]}), wrong {sum(v for (t, w), v in wrong.items() if t == b)}")
    if wrong:
        print("  confusions:", wrong.most_common(8))
