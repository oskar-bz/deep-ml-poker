"""Bet-size tells: bettor's current hand strength (percentile vs all combos on the board) by bet size."""
import glob, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
from poker_engine.range_engine import combo_scores, COMBO_INDEX
from pokerbot_sdk import parse_card
HIGHER_BETTER = None
def strength(hole, board):
    b = [parse_card(c) for c in board]; h = [parse_card(c) for c in hole]
    sc, valid = combo_scores(b)
    me = sc[COMBO_INDEX[h[0], h[1]]]
    v = sc[valid]
    return float((v < me).mean() + 0.5 * (v == me).mean()) if HIGHER_BETTER else float((v > me).mean() + 0.5 * (v == me).mean())
# orientation check: quads vs nothing
global_sc, _ = combo_scores([parse_card(c) for c in ("2c", "2d", "2h")])
HIGHER_BETTER = global_sc[COMBO_INDEX[parse_card("2s"), parse_card("Ac")]] > global_sc[COMBO_INDEX[parse_card("7s"), parse_card("8c")]]
D = defaultdict(list)
TOPS = ["mw", "Shresth Rana", "Daksh Malhan", "thearly man", "Henrik Gombos", "Lorenzo P.", "Kashyap Patel", "Sujay",
        "krishnah37", "Scott Middlecamp", "Vaibhaav Tiwari", "Sam Thomas", "mohansriram", "Vedant Bhardwaj",
        "vidvuds calitis", "Elias Marlowe"]
for f in sorted(glob.glob("data/r1/r1-*.json"))[:400]:
    g = json.load(open(f, encoding="utf-8"))["game"]
    nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for h in g["hands"]:
        prev = 0
        for st in range(4):
            com = defaultdict(int); bet_seen = False
            for a in (x for x in h["a"] if x[0] == st):
                s, c, amt = a[1], a[2], a[3]
                if st > 0 and nm[s] in TOPS and c in "XBRA" and len(h["board"]) >= 3:
                    board = h["board"][: {1: 3, 2: 4, 3: 5}[st]]
                    pot = prev + sum(com.values())
                    stg = strength(h["hole"][s], board)
                    if c == "X":
                        D[(nm[s], "check")].append(stg)
                    elif not bet_seen:
                        frac = (amt - com[s]) / max(1, pot)
                        k = "allin" if c == "A" else ("b<.4" if frac < 0.4 else ("b.4-.7" if frac < 0.7 else ("b.7-1.1" if frac < 1.1 else "b>1.1")))
                        D[(nm[s], k)].append(stg)
                    else:
                        D[(nm[s], "raise" if c == "R" else "r-allin")].append(stg)
                if c in "BRA":
                    bet_seen = True
                if c in "CBRAb" and amt:
                    com[s] = max(com[s], amt)
            prev += sum(com.values())
K = ["check", "b<.4", "b.4-.7", "b.7-1.1", "b>1.1", "allin", "raise", "r-allin"]
print("mean hand-strength percentile (0..1) of the actor; n in brackets")
print(f"{'bot':<17}" + "".join(f"{k:>14}" for k in K))
for n in TOPS:
    row = ""
    for k in K:
        L = D[(n, k)]
        row += f"{np.mean(L):8.2f}({len(L):4d})" if len(L) >= 15 else f"{'-':>14}"
    print(f"{n[:16]:<17}{row}")
