"""Fit empirical policy models of round-1 bots from data/r1 replays -> data/r1_models.json.

Preflop: P(action | n_raises_before (0,1,2+), depth bucket, hand-strength decile) + sizes.
Postflop: first-in bet freq, fold/raise-vs-bet freq split by made-hand class, bet size / pot.
"""
import glob, json, sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, ".")
from poker_engine.preflop_multiway import MULTIWAY_EQ

R = "23456789TJQKA"
off = min(k[0] for k in MULTIWAY_EQ)
order = sorted(MULTIWAY_EQ, key=lambda k: -MULTIWAY_EQ[k][0])
TOP, cum = {}, 0
for k in order:
    w = 6 if k[0] == k[1] else (4 if k[2] else 12)
    TOP[k] = (cum + w / 2) / 1326
    cum += w


def top_of(h):
    r = sorted((R.index(c[0]) for c in h), reverse=True)
    return TOP[(r[0] + off, r[1] + off, h[0][1] == h[1][1] and r[0] != r[1])]


def depth(bbs):
    return 0 if bbs < 15 else (1 if bbs < 40 else 2)


def made_class(hole, board):
    """0 = nothing, 1 = pair (any), 2 = two pair+ (rough, rank-count based) / flush/straight ignored."""
    ranks = [c[0] for c in hole + board]
    cnt = sorted((ranks.count(x) for x in set(ranks)), reverse=True)
    hole_r = {c[0] for c in hole}
    board_r = [c[0] for c in board]
    if cnt[0] >= 3 or (cnt[0] == 2 and cnt[1] == 2 and any(x in board_r for x in hole_r)):
        return 2 if any(ranks.count(x) >= 2 for x in hole_r) else 1
    if any(x in board_r for x in hole_r) or hole[0][0] == hole[1][0]:
        return 1
    return 0


def main(names):
    pre = defaultdict(lambda: [[0, 0, 0, 0] for _ in range(10)])   # key (name, nr, d) -> deciles -> F,C,R,A counts
    sizes = defaultdict(list)                                       # (name, nr, d) -> raise-to / (bb or cur bet)
    post = defaultdict(lambda: [0.0] * 3)
    for f in sorted(glob.glob("data/r1/r1-*.json")):
        g = json.load(open(f, encoding="utf-8"))["game"]
        nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
        for h in g["hands"]:
            bb = h["b"][1]
            nr, cur, jam1 = 0, bb, False
            street_bet = defaultdict(int)
            bet_on = {}
            for a in h["a"]:
                st, s, c, amt = a[0], a[1], a[2], a[3]
                n = nm[s]
                if st == 0:
                    if c == "b":
                        continue
                    if n in names:
                        d = depth(h["s"][s] / bb)
                        dec = min(9, int(top_of(h["hole"][s]) * 10))
                        key = (n, "j" if (nr == 1 and jam1) else min(nr, 2), d)
                        i = {"F": 0, "X": 1, "C": 1, "R": 2, "B": 2, "A": 3}[c]
                        pre[key][dec][i] += 1
                        if c == "R":
                            sizes[key].append(amt / (bb if nr == 0 else cur))
                    if c in "RA" and amt > cur:
                        if nr == 0:
                            jam1 = c == "A"
                        nr += 1; cur = amt
                    continue
                if n not in names:
                    if c in "BRA":
                        bet_on[st] = 1
                    continue
                # postflop stats
                board = h["board"][: {1: 3, 2: 4, 3: 5}[st]]
                mc = made_class(h["hole"][s], board)
                P = post
                if st not in bet_on:
                    P[(n, "first", mc)][0] += 1
                    if c in "BA":
                        P[(n, "first", mc)][1] += 1
                        if c == "B" and h.get("pots") is None:
                            pass
                else:
                    P[(n, "face", mc)][0] += 1
                    P[(n, "face", mc)][1] += c == "F"
                    P[(n, "face", mc)][2] += c in "RA"
                if c in "BRA":
                    bet_on[st] = 1
    out = {}
    for n in names:
        m = {"pre": {}, "size": {}, "post": {}}
        for nr in (0, 1, 2, "j"):
            for d in range(3):
                key = (n, nr, d)
                if key not in pre:
                    continue
                rows = []
                for dec in range(10):
                    cts = pre[key][dec]
                    t = sum(cts)
                    # light smoothing toward neighbouring decile behaviour
                    rows.append([(x + 0.25) / (t + 1.0) for x in cts])
                m["pre"][f"{nr}{d}"] = rows
                if sizes[key]:
                    m["size"][f"{nr}{d}"] = median(sizes[key])
        for kind in ("first", "face"):
            for mc in range(3):
                v = post.get((n, kind, mc))
                if v and v[0] > 0:
                    m["post"][f"{kind}{mc}"] = [v[1] / v[0], v[2] / v[0], v[0]]
        out[n] = m
    json.dump(out, open("data/r1_models.json", "w"), indent=0)
    for n in names:
        print(n, {k: round(v, 2) for k, v in out[n]["size"].items()},
              {k: [round(x, 2) for x in v[:2]] for k, v in out[n]["post"].items()})


if __name__ == "__main__":
    main(["mw", "Shresth Rana", "Daksh Malhan", "thearly man", "Henrik Gombos", "Lorenzo P.", "Kashyap Patel",
          "Sujay", "krishnah37", "Scott Middlecamp", "Vaibhaav Tiwari", "Sam Thomas", "mohansriram",
          "Vedant Bhardwaj", "vidvuds calitis", "Elias Marlowe", "Oskar Behzadpour"])
