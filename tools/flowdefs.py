"""Shared hand categorisation for real replays and simulated hands.
acts: list of (street, seat, kind, chips_put_in_total_this_street) with kind in fold/check/call/raise/blind."""
def categorize(acts, me, start_stack, contrib, delta_bb):
    my = [a for a in acts if a[1] == me and a[2] != "blind"]
    if not my:
        return None
    maxst = max(a[0] for a in acts)
    pre = [a for a in acts if a[0] == 0 and a[2] != "blind"]
    out = {}
    if contrib >= start_stack:
        out["cat"] = "allin_pre" if maxst == 0 or max(a[0] for a in my) == 0 else "allin_post"
    elif my[0][0] == 0 and my[0][2] == "fold":
        out["cat"] = "pre_fold"
    elif maxst == 0:
        out["cat"] = "pre_raise_end" if any(a[2] == "raise" for a in my) else "pre_other"
    elif max(a[0] for a in my) >= 1:
        out["cat"] = "postflop"
    else:
        out["cat"] = "other"
    # open-raise: our first preflop action is the first raise and not all-in
    first_raise = next((a for a in pre if a[2] == "raise"), None)
    if first_raise is not None and first_raise[1] == me and my[0] is first_raise and contrib < start_stack:
        after = pre[pre.index(first_raise) + 1:]
        out["open"] = 1
        nb = len({a[1] for a in after if a[1] != me})
        out["pb"] = "1" if nb <= 1 else ("2-3" if nb <= 3 else "4+")
        out["open_steal"] = int(all(a[2] == "fold" for a in after) and maxst == 0)
        out["open_3bet"] = int(any(a[2] == "raise" for a in after if a[1] != me))
    out["d"] = delta_bb
    return out
