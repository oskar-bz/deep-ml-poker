# CELL 2 (TEMPORARY - DELETE THIS CELL BEFORE PRESSING SUBMIT)
# Checks the real SDK against the adapter's assumptions. Paste the full output back to Claude.
from pokerbot_sdk import play_local


class Probe(MyBot):
    calls = 0
    problems = []

    def act(self, state):
        Probe.calls += 1
        if Probe.calls <= 2:
            print("=== RAW STATE #%d ===" % Probe.calls)
            try:
                print({k: v for k, v in vars(state).items() if not k.startswith("__")})
            except TypeError:
                print([a for a in dir(state) if not a.startswith("__")])
            print("type:", type(state), "| history sample:", list(getattr(state, "history", []) or [])[:6])
        t0 = time.perf_counter()
        action = super().act(state)
        ms = (time.perf_counter() - t0) * 1000
        d = self._last or {}
        if d:
            h = d["hero_idx"]
            stack = getattr(state, "stack", None)
            checks = {
                "hero stack == state.stack": stack is None or d["stacks"][h] == stack,
                "bets[hero]+to_call == current_bet": d["street_bets"][h] + d["to_call"] == getattr(state, "current_bet", max(d["street_bets"])) or d["to_call"] >= d["stacks"][h],
                "pot == sum(contributions)": sum(d["hand_contributions"]) == d["pot"],
                "hero active": d["active_seats"][h],
            }
            for k, ok in checks.items():
                if not ok and len(Probe.problems) < 8:
                    Probe.problems.append((k, Probe.calls, {"hero": h, "button": d["button_idx"], "pot": d["pot"],
                                                            "contrib": d["hand_contributions"], "bets": d["street_bets"]}))
            if Probe.calls <= 2:
                print("ADAPTER INFERRED: hero", h, "button", d["button_idx"], "sb/bb", d["sb_seat"], d["bb_seat"],
                      "hand#", d["hand_number"], "raise_as_total", self._raise_as_total, "| action", action, "| %.3f ms" % ms)
        return action


play_local(Probe(), tables=3)
print("decisions:", Probe.calls)
print("PROBLEMS:", Probe.problems or "none")
