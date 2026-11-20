"""Feed REAL tournament hands (data/rN replays) to a built bot in the platform's formats:
a State-like object at each of our decisions and a result dict at each hand end (format seen in the
round-2 probe). Reports exceptions inside the core and what the core learned about each opponent.

python tools/replay_feed.py BUILD.py data/r3 [games] [--no-meta]
  --no-meta : State without seat / hand_number attributes (adapter must infer them)
"""
import glob
import importlib.util
import json
import sys
import traceback

sys.path.insert(0, ".")
ME = "Oskar Behzadpour"
ST = ["preflop", "flop", "turn", "river"]
NAMES = {"F": "fold", "X": "check", "C": "call", "B": "bet", "R": "raise", "b": "blind"}


class PState:
    """Attribute bag like the platform's State (vars() returns {} there, so use slots-free getattr only)."""
    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


def platform_action(code, street, bet_seen):
    if code == "A":
        return ("bet" if (street > 0 and not bet_seen) else "raise"), True
    return NAMES[code], False


AGREE = {"n": 0, "same": 0, "diffs": []}
SHOWN_ONLY = "--shown-only" in sys.argv


def _kind(code):
    return {"F": "fold", "X": "check", "C": "call", "B": "raise", "R": "raise", "A": "raise"}[code]


def feed_game(S, g, seat, meta=True, errs=None, hand_cap=None, call_end=True):
    bot = S.MyBot()
    core = bot.core
    names = [p["name"].replace(" (copy)", "") for p in g["players"]]
    for hidx, h in enumerate(g["hands"]):
        if hand_cap and hidx >= hand_cap:
            break
        if h["s"][seat] <= 0:
            continue
        bb = h["b"][1]; sbv = h["b"][0]
        n = 8
        stacks = list(h["s"])
        hist = []
        pot = 0
        street_bets = [0] * n
        cur_street = 0
        bet_seen = False
        board_all = h["board"]
        acted = False
        for a in h["a"]:
            st, s, code, amt = a[0], a[1], a[2], a[3]
            if st != cur_street:
                cur_street = st; street_bets = [0] * n; bet_seen = False
            if s == seat and code != "b":
                # our decision point: build the state BEFORE applying our action
                cur = max(street_bets)
                to_call = min(stacks[seat], cur - street_bets[seat])
                last_raise = bb
                min_raise_to = cur + max(bb, last_raise) if cur > 0 else bb
                legal = ["fold", "call"] if to_call > 0 else ["check"]
                if stacks[seat] > to_call:
                    legal.append("raise")
                nb = {0: 0, 1: 3, 2: 4, 3: 5}[st]
                kw = dict(hole=list(h["hole"][seat]), board=list(board_all[:nb]), street=ST[st], to_call=to_call,
                          pot=pot, min_raise=min(street_bets[seat] + stacks[seat], min_raise_to),
                          max_raise=street_bets[seat] + stacks[seat], current_bet=cur, legal=legal,
                          stacks=list(stacks), bets=list(street_bets), big_blind=bb, small_blind=sbv,
                          level=min(7, hidx // 25), stack_m=stacks[seat] / max(1, 1.5 * bb),
                          opponents=sum(1 for x in range(n) if x != seat and h["s"][x] > 0),
                          history=[dict(e) for e in hist])
                if meta:
                    kw.update(seat=seat, hand_number=hidx, button=h["btn"])
                try:
                    act = bot.act(PState(**kw))
                    if st == 0:
                        mine = act.kind if act.kind != "all_in" else "raise"
                        real = _kind(code)
                        if real == "raise" and code == "A" and amt <= max(street_bets):
                            real = "call"
                        AGREE["n"] += 1
                        if mine == real or (mine in ("check", "fold") and real in ("check", "fold")):
                            AGREE["same"] += 1
                        elif len(AGREE["diffs"]) < 400:
                            AGREE["diffs"].append((hidx, real, mine, h["hole"][seat], round(to_call / bb, 1)))
                except Exception:
                    errs.append(("act", traceback.format_exc()))
                acted = True
            # apply the real action
            name, allin = platform_action(code, st, bet_seen) if code != "A" else platform_action("A", st, bet_seen)
            if code == "A":
                # all-in: amount = raise-to total for the street (or a call of the current bet)
                total = amt
                if total <= max(street_bets):
                    name = "call"
            if code in "bCBRA":
                add = max(0, amt - street_bets[s])
                add = min(add, stacks[s])
                street_bets[s] += add; stacks[s] -= add; pot += add
            if code in "BRA" and st > 0:
                bet_seen = True
            hist.append({"street": ST[st], "seat": s, "action": name, "amount": amt if code != "X" and code != "F" else 0,
                         "all_in": bool(allin or (code in "CBRA" and stacks[s] == 0))})
        if not acted:
            continue
        nb_end = len(board_all)
        if SHOWN_ONLY:
            folded = {a[1] for a in h["a"] if a[2] == "F"}
            live_end = [i for i in range(n) if h["s"][i] > 0 and i not in folded]
            showdown = set(live_end) if len(live_end) >= 2 else set()
            holes = [list(x) if (h["s"][i] > 0 and (i in showdown or i == seat)) else [] for i, x in enumerate(h["hole"])]
        else:
            holes = [list(x) if h["s"][i] > 0 else [] for i, x in enumerate(h["hole"])]
        result = {"hand_id": hidx, "button": h["btn"], "blinds": [sbv, bb], "start_stacks": list(h["s"]),
                  "hole": holes,
                  "board": list(board_all[:nb_end]), "actions": hist,
                  "pots": [], "shown": {}, "hand_names": {}, "deltas": list(h["d"]), "seat": seat}
        try:
            if call_end:
                bot.on_hand_end(result)
        except Exception:
            errs.append(("end", traceback.format_exc()))
    return core, names


def main():
    path, folder = sys.argv[1], sys.argv[2]
    ngames = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 20
    meta = "--no-meta" not in sys.argv
    call_end = "--no-end" not in sys.argv
    spec = importlib.util.spec_from_file_location("built", path)
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    core_errs = []

    class Core(S.ContestCore):
        def act(self, d):
            try:
                return super().act(d)
            except Exception:
                core_errs.append(("core.act", traceback.format_exc())); raise

        def on_hand_end(self, i, s):
            try:
                return super().on_hand_end(i, s)
            except Exception:
                core_errs.append(("core.on_hand_end", traceback.format_exc())); raise

    S.MyBot.core_factory = Core
    errs = []
    files = sorted(glob.glob(folder + "/*.json"))[:ngames]
    agg = {}
    for f in files:
        g = json.load(open(f, encoding="utf-8"))["game"]
        seats = [i for i, p in enumerate(g["players"]) if p["name"] == ME]
        for seat in seats:
            core, names = feed_game(S, g, seat, meta, errs, None, call_end)
            for s in range(8):
                if s == seat:
                    continue
                a = agg.setdefault(names[s], [0, 0, 0, 0, 0, 0])
                a[0] += core.faced_steal[s]; a[1] += core.threebet_count[s]
                a[2] += len(core.tb_deep[s]); a[3] += core.faced_jam[s]; a[4] += core.faced_3bet[s]
                a[5] += 1
    print("adapter-level exceptions:", len(errs), "| core exceptions:", len(core_errs))
    print("PREFLOP agreement with the real tournament decisions: %d / %d = %.2f%%" % (
        AGREE["same"], AGREE["n"], 100.0 * AGREE["same"] / max(1, AGREE["n"])))
    from collections import Counter
    print("  differing (real -> fed):", Counter((d[1], d[2]) for d in AGREE["diffs"]).most_common(6))
    for kind, tb in (errs + core_errs)[:2]:
        print(kind); print(tb[-1500:])
    print(f"{'opponent':<18}{'faced_open':>11}{'3bets':>7}{'deep3b hole obs':>17}{'faced_jam':>10}{'faced_3bet':>11}  (sum over tables)")
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1][5]):
        print(f"{k[:17]:<18}{a[0]:11.0f}{a[1]:7.0f}{a[2]:17.0f}{a[3]:10.0f}{a[4]:11.0f}")


if __name__ == "__main__":
    main()
