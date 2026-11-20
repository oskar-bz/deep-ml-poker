# Contest-hour runbook (60 min, 3 submissions)

Files (all in `contest/`):
- `cell1_submission_bot.py`: the bot (ApexPlus core). Must be the FIRST cell, pasted whole.
- `cell2_probe.py`: checks the real SDK against our assumptions. DELETE before submitting.
- `cell3_test.py`: `play_local(MyBot(), tables=20)` (stripped on submit anyway).
- `submission_fallback_apexbot.py`: same bot without the numpy best-response layer
  (lower latency). Use if think time is too high or numpy misbehaves.

## Timeline
| min | step |
|---|---|
| 0-3 | Start session. Delete the notebook's starter cells. Paste cell 1, run it (should be silent, <1 s). |
| 3-6 | Paste + run cell 2. Copy the FULL output to Claude. |
| 6-20 | Claude patches field names / conventions if PROBLEMS are reported (edits in `bots/sdk_adapter.py`, rebuild). Re-run cells 1+2 until `PROBLEMS: none`. |
| 20-25 | Run cell 3. Check avg points and think-time stats (worst case must be well under 1 ms; target < 0.6 ms). |
| 25 | Delete cell 2. **Submission 1** = ApexPlus. |
| 25-45 | Read validation result (bb/100 vs house bots, avg/max ms, timeouts). |
| 45 | Only if submission 1 failed validation or shows timeouts: **Submission 2** = fallback file. |
| -- | Keep submission 3 in reserve. **The LAST submission that passes its demo tables is the one that enters** (official rules), so never end on an experiment. Don't resubmit to re-roll demo variance: demo tables are 8 x 10 s and only set the entry leaderboard; the tournament starts from scratch. |

## Facts from the official rules
- Blinds 10/20, 20/40, 30/60, 50/100, 75/150, 100/200, 150/300, 200/400 (every 25 hands).
- A crashed process is folded for the rest of the table; exceptions/timeouts = check/fold.
- Between bracket rounds: one more 60-min session + one resubmission (prepare improvements offline before it).

## Decision rules
- FIRST look at the starter bot the notebook opens with:
  - long version (bets `state.current_bet + state.pot // 2`, raises `state.min_raise * 2`)
    -> main bot = `cell1_submission_bot.py` (ApexPlus exploits exactly this bot, ~+0.2 pts).
  - short version (`Action.raise_to(state.min_raise)`, postflop strength 0.5)
    -> main bot = `submission_fallback_apexbot.py` (ApexPlus gains nothing there, only latency risk).
- Probe shows `PROBLEMS: none` and play_local worst think time < 0.6 ms -> submit cell 1 as is.
- Worst think time 0.6-0.9 ms -> add `ContestCore.BR_SAMPLES = 90` at the end of cell 1.
- Worst > 0.9 ms or any timeouts -> use the fallback file.
- Any exception / crash in cell 1 -> paste the traceback to Claude; fallback file if not fixed in 10 min.
