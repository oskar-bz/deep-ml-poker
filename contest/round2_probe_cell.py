# OPTIONAL CELL (run after the bot cell, DELETE BEFORE SUBMITTING)
# Checks that on_hand_end(result) carries per-seat chip deltas: the shove/re-shove exploits learn
# opponents' call/fold rates from them. Expect "deltas found: True".
from pokerbot_sdk import play_local


class ResultProbe(MyBot):
    seen = 0

    def on_hand_end(self, result):
        if ResultProbe.seen < 2:
            ResultProbe.seen += 1
            try:
                print("RESULT:", {k: v for k, v in vars(result).items() if not k.startswith("__")})
            except TypeError:
                print("RESULT type", type(result), result if isinstance(result, dict) else
                      [a for a in dir(result) if not a.startswith("__")])
            d = _attr(result, ("deltas", "delta", "chip_deltas", "net"), None)
            print("deltas found:", d is not None, d)
        super().on_hand_end(result)


play_local(ResultProbe(), tables=1)
