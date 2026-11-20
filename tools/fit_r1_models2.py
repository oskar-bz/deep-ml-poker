"""Version-2 empirical policy models of all round-1 bots -> data/r1_models2.json.

Preflop  ctx (nr, depth, pos)      x hand decile                 -> F, C, R, A   (+ sizes)
Postflop first-in ctx (street, role, multiway) x strength decile x draw -> X, B<.4, B.4-.7, B>=.7, A (+ sizes)
Postflop facing  ctx (street, size bucket / raise, multiway) x strength decile x draw -> F, C, R, A

Strength = current made-hand percentile vs all live combos on the board (range_engine.combo_scores).
Sparse cells back off: bot-specific -> bot coarse -> population.
"""
import glob, json, sys
from collections import defaultdict
from statistics import median

import numpy as np

sys.path.insert(0, ".")
from poker_engine.preflop_multiway import MULTIWAY_EQ
from poker_engine.range_engine import combo_scores, COMBO_INDEX
from pokerbot_sdk import parse_card

RK = "23456789TJQKA"
_off = min(k[0] for k in MULTIWAY_EQ)
_order = sorted(MULTIWAY_EQ, key=lambda k: -MULTIWAY_EQ[k][0])
TOP, _cum = {}, 0
for _k in _order:
    _w = 6 if _k[0] == _k[1] else (4 if _k[2] else 12)
    TOP[(_k[0] - _off, _k[1] - _off, _k[2])] = (_cum + _w / 2) / 1326
    _cum += _w


def pre_decile(hole):
    r = sorted((RK.index(c[0]) for c in hole), reverse=True)
    top = TOP[(r[0], r[1], hole[0][1] == hole[1][1] and r[0] != r[1])]
    return min(9, int((1.0 - top) * 10))          # 9 = best


def pre_decile_frac(hole):
    r = sorted((RK.index(c[0]) for c in hole), reverse=True)
    v = (1.0 - TOP[(r[0], r[1], hole[0][1] == hole[1][1] and r[0] != r[1])]) * 10
    d = min(9, int(v))
    return d, min(0.999, v - d)


_sc_cache = {}


def strength(hole_i, board_i):
    key = tuple(board_i)
    v = _sc_cache.get(key)
    if v is None:
        sc, valid = combo_scores(board_i)
        v = (sc, np.sort(sc[valid]))
        if len(_sc_cache) > 4000:
            _sc_cache.clear()
        _sc_cache[key] = v
    sc, srt = v
    me = sc[COMBO_INDEX[hole_i[0], hole_i[1]]]
    lo = np.searchsorted(srt, me, "left"); hi = np.searchsorted(srt, me, "right")
    return float((lo + 0.5 * (hi - lo)) / len(srt))   # higher score = better (checked below)


def has_draw(hole_i, board_i):
    if len(board_i) >= 5:
        return 0
    cards = list(hole_i) + list(board_i)
    for s in range(4):
        if sum(1 for c in cards if c & 3 == s) == 4 and any(c & 3 == s for c in hole_i):
            return 1
    ranks = {c >> 2 for c in cards}
    if 12 in ranks:
        ranks.add(-1)
    hr = {c >> 2 for c in hole_i}
    for lo in range(-1, 9):
        w = set(range(lo, lo + 4))
        if w <= ranks and (w & hr or (12 in hr and lo == -1)):
            return 1
    return 0


def depth(bbs):
    return 0 if bbs < 15 else (1 if bbs < 25 else (2 if bbs < 40 else 3))


def pos_bucket(behind):
    return 0 if behind == 0 else (1 if behind == 1 else (2 if behind <= 3 else 3))


def size_bucket(frac):
    return 0 if frac < 0.4 else (1 if frac < 0.7 else (2 if frac < 1.1 else 3))


PRE_POP_W = 1.0   # preflop backoff weights (lower = more bot-specific)
PRE_CO_W = 1.0
PRE_A = {"F": 0, "X": 1, "C": 1, "R": 2, "B": 2, "A": 3}


def main(folder="data/r1", outfile="data/r1_models2.json"):
    pre = defaultdict(lambda: np.zeros(4))
    pre_size = defaultdict(list)
    fi = defaultdict(lambda: np.zeros(5))
    fi_size = defaultdict(list)
    fa = defaultdict(lambda: np.zeros(4))
    fa_rsize = defaultdict(list)
    games = defaultdict(int)
    files = sorted(glob.glob(folder + "/*.json"))
    for gi, f in enumerate(files):
        g = json.load(open(f, encoding="utf-8"))["game"]
        nm = [p["name"].replace(" (copy)", "") for p in g["players"]]
        for n in nm:
            games[n] += 1
        for h in g["hands"]:
            bb = h["b"][1]
            dealt = [x for x in range(8) if h["s"][x] > 0]
            hole_i = {x: (parse_card(h["hole"][x][0]), parse_card(h["hole"][x][1])) for x in dealt}
            blinds = [a[1] for a in h["a"] if a[2] == "b"]
            bb_seat = blinds[1] if len(blinds) > 1 else blinds[0]
            board_all = [parse_card(c) for c in h["board"]]
            folded = set(); allin = set()
            prev = 0
            last_aggr = -1
            for st in range(4):
                com = defaultdict(int)
                nr, cur, jam1 = 0, (bb if st == 0 else 0), False
                last_allin = False
                street_aggr = -1
                acted = set()
                cur_bet_frac = 0.0
                for a in (x for x in h["a"] if x[0] == st):
                    s, c, amt = a[1], a[2], a[3]
                    n = nm[s]
                    if c == "b":
                        com[s] = max(com[s], amt)
                        continue
                    live = [x for x in dealt if x not in folded]
                    mw = 1 if len(live) > 2 else 0
                    if st == 0:
                        d = depth(h["s"][s] / bb)
                        dec = pre_decile(h["hole"][s])
                        nk = "j" if (nr == 1 and jam1) else str(min(nr, 3))
                        if nk == "1" and cur > 2.6 * bb:
                            nk = "1L"
                        if nr <= 1:
                            # players still to act after s up to bb
                            behind = 0
                            if s != bb_seat:
                                x = s
                                for _ in range(8):
                                    x = (x + 1) % 8
                                    if x in live and x != s:
                                        behind += 1
                                    if x == bb_seat:
                                        break
                            p = pos_bucket(behind)
                            if nr == 1 and s == bb_seat:
                                p = 0
                        else:
                            p = 0
                        pre[(n, nk, d, p, dec)][PRE_A[c]] += 1
                        if c == "R":
                            pre_size[(n, nk, d)].append(amt / (bb if nr == 0 else max(cur, bb)))
                    elif len(board_all) >= 3:
                        board = board_all[: {1: 3, 2: 4, 3: 5}[st]]
                        sg = strength(hole_i[s], board)
                        dec = min(9, int(sg * 10))
                        dr = has_draw(hole_i[s], board)
                        pot = prev + sum(com.values())
                        if nr == 0:
                            role = 1 if last_aggr == s else 0
                            if c == "X":
                                i = 0
                            elif c == "A":
                                i = 4
                            else:
                                frac = (amt - com[s]) / max(1, pot)
                                i = 1 + min(2, size_bucket(frac))
                                fi_size[(n, i)].append(frac)
                            fi[(n, st, role, mw, dec, dr)][i] += 1
                        else:
                            sb = 5 if last_allin else (4 if nr >= 2 else size_bucket(cur_bet_frac))
                            i = {"F": 0, "C": 1, "R": 2, "A": 3}.get(c)
                            if i is not None:
                                fa[(n, st, sb, mw, dec, dr)][i] += 1
                                if c == "R":
                                    fa_rsize[n].append(amt / max(1, cur))
                    if c == "F":
                        folded.add(s)
                    if c in "BRA" and amt > cur:
                        if st > 0 and nr == 0:
                            cur_bet_frac = (amt - com[s]) / max(1, prev + sum(com.values()))
                        if nr == 0:
                            jam1 = c == "A"
                        last_allin = c == "A"
                        nr += 1; cur = amt; street_aggr = s
                    if c in "CBRA" and amt:
                        com[s] = max(com[s], amt)
                prev += sum(com.values())
                if street_aggr >= 0:
                    last_aggr = street_aggr
                else:
                    last_aggr = -1 if st > 0 else last_aggr
        if gi % 100 == 0:
            print("game", gi, flush=True)

    names = sorted(games)
    out = {"_games": dict(games), "pre": {}, "pre_size": {}, "fi": {}, "fi_size": {}, "fa": {}, "fa_rsize": {}}

    def norm(v):
        return (v / v.sum()).round(4).tolist() if v.sum() > 0 else None

    # ---------- preflop: (name,nk,d,p,dec) -> back off to (name,nk,d,dec) -> (POP,nk,d,p,dec)
    pop_pre = defaultdict(lambda: np.zeros(4)); coarse_pre = defaultdict(lambda: np.zeros(4))
    for (n, nk, d, p, dec), v in pre.items():
        pop_pre[(nk, d, p, dec)] += v; coarse_pre[(n, nk, d, dec)] += v
    for n in names:
        tab = {}
        for nk in ("0", "1", "1L", "j", "2", "3"):
            for d in range(4):
                for p in range(4 if nk in ("0", "1", "1L", "j") else 1):
                    rows = []
                    for dec in range(10):
                        pp = pop_pre[(nk, d, p, dec)] + 0.25
                        pp = pp / pp.sum()
                        cc = coarse_pre[(n, nk, d, dec)]
                        pc = (cc + PRE_POP_W * pp) / (cc.sum() + PRE_POP_W)
                        sp = pre[(n, nk, d, p, dec)]
                        rows.append(((sp + PRE_CO_W * pc) / (sp.sum() + PRE_CO_W)).round(4).tolist())
                    tab[f"{nk}{d}{p}"] = rows
        out["pre"][n] = tab
        out["pre_size"][n] = {f"{nk}{d}": float(median(L)) for (m, nk, d), L in pre_size.items() if m == n and len(L) >= 3}

    # ---------- postflop first-in
    pop_fi = defaultdict(lambda: np.zeros(5)); co_fi = defaultdict(lambda: np.zeros(5))
    for (n, st, role, mw, dec, dr), v in fi.items():
        pop_fi[(st, role, mw, dec, dr)] += v; co_fi[(n, role, dec, dr)] += v
    pop_fa = defaultdict(lambda: np.zeros(4)); co_fa = defaultdict(lambda: np.zeros(4))
    for (n, st, sb, mw, dec, dr), v in fa.items():
        pop_fa[(st, sb, mw, dec, dr)] += v; co_fa[(n, sb, dec, dr)] += v
    for n in names:
        t1 = {}
        for st in (1, 2, 3):
            for role in (0, 1):
                for mw in (0, 1):
                    rows = []
                    for dec in range(10):
                        for dr in (0, 1):
                            pp = pop_fi[(st, role, mw, dec, dr)] + 0.2
                            pp = pp / pp.sum()
                            cc = co_fi[(n, role, dec, dr)]
                            pc = (cc + 5.0 * pp) / (cc.sum() + 5.0)
                            sp = fi[(n, st, role, mw, dec, dr)]
                            rows.append(((sp + 4.0 * pc) / (sp.sum() + 4.0)).round(4).tolist())
                    t1[f"{st}{role}{mw}"] = rows
        out["fi"][n] = t1
        t2 = {}
        for st in (1, 2, 3):
            for sb in range(6):
                for mw in (0, 1):
                    rows = []
                    for dec in range(10):
                        for dr in (0, 1):
                            pp = pop_fa[(st, sb, mw, dec, dr)] + 0.2
                            pp = pp / pp.sum()
                            cc = co_fa[(n, sb, dec, dr)]
                            pc = (cc + 5.0 * pp) / (cc.sum() + 5.0)
                            sp = fa[(n, st, sb, mw, dec, dr)]
                            rows.append(((sp + 4.0 * pc) / (sp.sum() + 4.0)).round(4).tolist())
                    t2[f"{st}{sb}{mw}"] = rows
        out["fa"][n] = t2
        out["fi_size"][n] = {str(i): float(median(fi_size[(n, i)])) for i in (1, 2, 3) if len(fi_size[(n, i)]) >= 3}
        out["fa_rsize"][n] = float(median(fa_rsize[n])) if len(fa_rsize[n]) >= 3 else 3.0
    json.dump(out, open(outfile, "w", encoding="utf-8"))
    print("bots", len(names))


if __name__ == "__main__":
    # orientation check for strength(): trips must beat nothing
    b = [parse_card(c) for c in ("2c", "2d", "7h")]
    assert strength((parse_card("2s"), parse_card("Ac")), b) > strength((parse_card("9s"), parse_card("Tc")), b), "flip"
    main(*sys.argv[1:3])
