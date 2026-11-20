# Deep-ML No-Limit Pokerbot Tournament - Project State

## 1. Rules (from the contest page / bundled JS, see `reference/starter_module.js`)
- 8-seat sit-and-go, 2,000 chips, blinds 10/20 rising every 25 hands, cap 200 hands
  (survivors split open places by ICM). Points 10 / 5 / 2, nothing below 3rd.
- **1 ms per decision**, measured around `act()`; late answer or error = check/fold.
  ~20 s start-up per table, one unscored warm-up decision. numpy + PyTorch (CPU), no network.
- Opponents are other entrants' bots; empty seats are filled with copies of entrants.
  Bracket rounds (1000 -> 500 -> 250 -> 125 -> 60 -> 30), ranked by round reached, then avg points.
- Public leaderboard (2026-11-03): 11 entrants, scores 7.26 down to 2.125 (= random baseline).

## 2. The real SDK API (extracted from the saved site)
```python
from pokerbot_sdk import Bot, Action
class MyBot(Bot):
    def act(self, state):   # -> Action.fold()/check()/call()/raise_to(total)/all_in()
        ...
    def on_hand_end(self, result): ...   # optional: deltas, stacks, shown cards
```
Documented `state` fields: `hole`, `board` ("As", "Td"), `street`, `to_call`, `pot`, `min_raise`,
`max_raise` (raise-to totals), `current_bet`, `legal`, `stacks`, `bets`, `big_blind`, `small_blind`,
`level`, `stack_m`, `opponents`, `history` (street, seat, action, amount), `pot_odds()`,
`preflop_rank()`, `preflop_equity()`, `made_hand()`, `equity(samples)` (~3 ms, too slow).
**Undocumented** (guessed, adapter has fallbacks): our seat, button, hand number, history amount
convention, action names. First thing in the contest notebook: `print(vars(state))` once.

## 3. Architecture
```
pokerbot_sdk.py            Local stand-in for the SDK (State/Action/Bot, play_local) + SDKBotDriver
                           (lets SDK bots sit at simulator tables).
bots/sdk_adapter.py        MyBot(Bot): SDK state -> canonical dict, (type, chips) -> Action.
                           Infers seat/button/hand#/amount convention if missing (exact in all
                           9 variants of tests/test_adapter_variants.py).
bots/apex_bot.py           Gemini's ApexBot (heuristic core, Bayesian profiler, ICM). Now has a
                           300 us Monte Carlo deadline and tunable P_* class attributes.
bots/apex_plus.py          ApexPlus = ApexBot + exact model of the STARTER_BOT family: identifies
                           starter clones by bet-size signature, verifies with showdowns, narrows
                           their range per action (numpy, all 1326 combos) and picks the postflop
                           best response by EV. Self-calibrates to the host CPU speed.
bots/sdk_field.py          Starter bot (verbatim), trained example, 3 typical starter edits.
poker_engine/range_engine.py      Vectorized 1326-combo scorer (~60 us/board).
poker_engine/preflop_multiway.py  Equity of 169 hands vs 1..7 opponents (tools/gen_preflop_multiway.py).
bench.py                   Parallel benchmark; fields: starter, example, sdkmix, realistic, llm,
                           round1, mirror. Hero overrides: "plus_sdk:P_X=1.2".
tools/diag.py              Leak finder (bb won/lost per line).  tools/tune.py: param tuner.
submission_builder.py      -> submission_bot.py (single file, core = ApexPlus).
```

## 4. Benchmarks (avg points/table, random = 2.125, ~±0.1)
| Field | ApexBot | ApexPlus |
|---|---|---|
| 7x starter | 7.25 | 7.45 |
| realistic (SDK bots + LLM-style) | 4.21 | 4.40 |
| sdkmix | 6.18 | 6.13-6.24 |
| trained example | 6.12 | 6.20 |
| mirror (3 copies of us) | 2.75 | 2.88 |
| llm (Gemini archetypes) | 2.94 | 2.85 (noise) |

Latency (single process): ApexBot mean ~60 us / max ~350 us; ApexPlus best-response ~330 us
mean / ~510 us max when it engages. Parameter tuning of ApexBot's constants found nothing
outside noise (tools/tune_log.txt).

## 5. Commands
```powershell
python bench.py --hero submission,apex_sdk --field all --tables 1000
python tests/test_adapter_fidelity.py ; python tests/test_adapter_variants.py
python submission_builder.py          # rebuild submission_bot.py
```

## 6. Contest-hour checklist
1. Cell 1: paste `submission_bot.py` (starts with `from __future__`, must be first).
2. Cell 2 (temporary): a bot whose act() does `print(vars(state)); print(state.history)` once.
   Check seat/button/hand_number names and history format against bots/sdk_adapter.py.
3. Cell 2: `play_local(MyBot(), tables=20)`; check think-time stats (worst case < ~600 us).
   If too slow on their hardware: set `ContestCore.BR_SAMPLES = 90` or make ContestCore
   subclass ApexBot instead of ApexPlus (rebuild).
4. Submit.

## 7. Round 1 result and round-2 resubmission (2026-11-16)
- Round 1 (500 tables): 3rd of 17 survivors, 3.83 pts/table (mw 4.28, Shresth 3.84). Per-table points
  have SD ~4, so the 500-table SE is ~0.18: places 2-5 were within noise of each other.
- Replays: `data/r1/` (561 games, all hole cards). Analysis tools: `tools/r1_analyze.py` (styles, faults,
  chip flow), `r1_deep.py`, `r1_ranges.py`, `r1_pressure.py` (bubble / call-vs-jam), `r1_busts.py`,
  `r1_betsize.py` (fold vs bet size), `r1_tells.py` (bet-size tells), `r1_facing.py` (our play vs bet sizes).
- Opponent models: `tools/fit_r1_models.py` -> `data/r1_models.json` (preflop by raises / facing-jam /
  depth / hand decile, postflop by made class, fold multipliers by bet size); `bots/sdk_field.py:R1ModelBot`;
  bench fields `r1top`, `r1mid`, `r1mw`, `r1mix`, `r1self`. Paired A/B: `python tools/paired.py A B fields n`.
- Changes in `contest/round2_submission_bot.py` (round-1 bot kept as `contest/round1_submission_bot.py`):
  - `P_W3_ON`: range of a 3-bettor from their observed 3-bet frequency; call jams / 4-bet jam vs wide ones.
  - `P_XJ_ON`: EV open-jam <= 20 bb vs players who rarely call jams; per-seat call rate learned from
    `on_hand_end` deltas after our jams (fallback: shown cards).
  - `P_STAB_ON` (+`P_STAB_FRAC=0.5`): EV half-pot stab when checked to heads-up vs observed over-folders.
  - Paired gains on model fields: roughly +0.1 (W3), +0.1..0.35 (XJ), +0.3..0.45 (STAB); neutral elsewhere.
- Known but not exploited: bet-size tells (large bets ~0.9 strength, small ~0.5); our measured leak from it
  is ~150-250 bb over 561 games (folding ahead to small bets, calling large bets behind).
- Unverified on the platform: the field names of the `on_hand_end` result (deltas). If absent, XJ keeps
  its 12% prior call rate (still jams, just without per-seat learning).
- v2 opponent models (`tools/fit_r1_models2.py` -> `data/r1_models2.json`, `bots/sdk_field.py:R2ModelBot`):
  all 33 round-1 bots; postflop by real hand-strength percentile / draw / role / bet size / multiway,
  preflop by position and open size, threshold-style action choice (NOISE=0.40). Calibration
  (`tools/calib.py hero`, `tools/flow_compare.py`): model-vs-real score correlation 0.92 (v1: ~0.66);
  our round-1 bot scores 3.7-3.9 vs the 16 finalists (real round 1: 3.83). Fields: `v2final` (16 finalists),
  `v2r1` (round-1 mix), `v2mirror` (finalists + a copy of us).
- Round-2 bot vs round-1 bot on v2 fields (paired, 3000 tables): v2final +0.40, v2r1 +0.25, v2mirror +0.44.
  Per change: re-steal +0.27..0.33, W3 +0.03..0.07, XJ +0.08..0.31, STAB +0.12..0.26.
- Last addition `P_XR_ON` (+`P_XR_MARGIN=1.5`, `P_XR_PRIOR=0.70`): EV re-jam over a single open at 8-25 bb
  (round-1: openers fold 87% to re-jams at 12-25 bb). v2final +0.63, v2r1 +0.27, v2mirror +0.24.
  Optional session check: `contest/round2_probe_cell.py` prints the on_hand_end result (expect deltas).
- Platform on_hand_end result (probe, play_local): dict with hand_id, button, blinds, start_stacks,
  hole (ALL players' hole cards, incl. folded), board, actions (full history), pots, shown, hand_names,
  deltas, seat. NEXT-ROUND OPPORTUNITY: if the tournament passes the same dict, every opponent's hole
  cards are visible after each hand -> exact per-opponent ranges per action (opens, 3-bets, calls, jams).

## 8. Round 2 result and round-3 prep (2026-11-17)
- Round 2 (834 tables): 1st, 4.23 pts/table (mw 4.21, Daksh 3.20). Round 3 field: us, mw, Daksh, Shresth,
  Henrik, Kashyap, Lorenzo, Vaibhaav, krishnah37. Replays: `data/r2/` (600 games).
- Models refit on round 2: `data/r2_models2.json` (depth buckets <15/15-25/25-40/40+, separate facing-4-bet
  context, lighter preflop backoff), bench fields `v3`, `v3mirror`; `F4_REAL` pins facing-4-bet folds to
  measured values. Calibration vs real round 2: corr 0.86.
- Leak vs mw (`tools/vs_opp.py data/r2 mw`): we open, mw 3-bets, we fold: -438 bb / 600 games. Deep, we fold
  even top-10% hands 72% of the time (base code only calls 3-bets costing <= 3.5 bb).
- Hole cards of all players are in the real on_hand_end result. Built (default OFF, not in the submission):
  per-seat deep 3-bet hand tracking (`tb_deep`, `f4_n/f4_fold`), `P_X4_ON` (4-bet light 3-bettors),
  `P_D3_ON` (equity-based deep 3-bet defence). Paired 3000 tables on v3: X4 +0.03, D3 -0.01/-0.10, both -0.05
  -> not adopted. Model mw's deep 3-bet range in sim is still tighter than real (unresolved).
- Decision: resubmit contest/round2_submission_bot.py unchanged.
- Model bug fixed (`R2ModelBot._pick`): the threshold noise was a weighted sum (0.6*rank + 0.4*rand), which
  is not uniform and under-selected rare actions (3-bets, folds). Now a mixture; per-situation 3-bet rates
  match reality (mw deep: sim 34/23/16/8% vs real 33/25/17/8%). Calibration after the fix: corr 0.70-0.82.
- With the corrected models: X4+D3 combined +0.17 (seed 1, 3000) but -0.02 (seed 7, 5000); mirror -0.05..-0.06
  -> still not adopted.
- Parameter sweep on v3 (16 variants, 2500 tables, `tools/sweep.sh`, log `data/sweep1.log`): all within noise;
  best candidate P_RS_SIZE=2.8 confirmed only +0.02/+0.07 on fresh seeds -> not adopted.
- Adversary fields (`bots/sdk_field.py`: Adv3BetBot, Adv4BetBot, AdvJamBot, AdvCallerBot, AdvRaiseBot; bench
  fields v3adv3 / v3adv4 / v3advjam / v3advcall / v3advraise). Light 3-bettors cost ~0.3 pts; facing a 4-bet
  we folded everything below top 4% (hardcoded) -> new `P_W4_ON` adaptive 4-bet defence.
- ROUND-3 SUBMISSION: `contest/round3_submission_bot.py` = round-2 bot + P_D3_ON (gated, P_D3_GATE) + P_X4_ON
  + P_W4_ON. Paired 3000: v3 -0.03 / 0.00, adv3 +0.45, adv4 +0.08, advjam +0.03, mirror +0.10.
  Platform-format result check (`tools/platform_format_check.py`): identical counters. Smoke test: no
  exceptions; unloaded think time avg 0.056 ms, max 0.295 ms. P_HB_ON (hole-card bluff learning) left off.
- Process history and method: `PROCESS.md`.

## 9. Cleanup and exact river (after round-3 submission)
- Feedback review: claimed bugs were not real or had no effect (D3/X4 order = intended polarised 4-bets;
  heads-up seat inference is correct for postflop; starter best-response layer never fires vs this field
  (0 of 814 postflop decisions); raise convention defaults to the platform's raise-to totals).
- Cleanup (no behaviour change, proven): removed the starter best-response layer from `bots/apex_plus.py`
  (old: `bots/legacy/apex_plus_starter_br.py`) and the off-switch hole-card bluff learning (P_HB) from
  `bots/apex_bot.py` (old: `bots/legacy/apex_bot_round3.py`); the builder now tree-shakes unreferenced
  definitions (`tools/treeshake.py`) and compiles the result. Build: 4680 -> 3847 lines.
  `tools/equiv.py A B tables adv` (record/replay with seeded RNG, no time budgets): 0 of 4950 decisions
  differ vs `contest/round3_submission_bot.py`.
- Exact heads-up river profile (`river_exact_profile`, switch P_RIVX_ON, time gate + import warm-up): matches
  40k-sample Monte Carlo within 0.01; ~130 us; A/B pooled ~+0.04 +- 0.03 but raises worst-case think time
  0.31 -> 0.47 ms locally -> left OFF (platform stretches tails 1.4-2.1x).
- `tools/build_check.py FILE` (exceptions + unloaded latency vs all adversaries) is the pre-submit check.
- Next-round candidate: `contest/round4_candidate_bot.py` (= round-3 decisions, cleaned build).

## 10. Round 3 result and round-4 (final) prep (2026-11-18)
- Round 3 (1250 tables): 2nd, 2.97 (mw 3.35, Daksh 2.52). Final: 8 bots, every table = all 8 finalists.
  Replays `data/r3/` (600 games). Kashyap had 8457 faults (times out -> forced folds).
- **Tournament on_hand_end reveals ONLY showdown hands (+ ours)**, not every hole card (the play_local probe
  did). Proven with `tools/replay_feed.py` (real r3 hands fed in platform format to the round-3 build;
  preflop decision agreement with the real tournament: all holes 98.42%, showdown-only 99.55%, no
  on_hand_end 94.14%). Consequences: the round-3 D3/X4 defences never fired in the tournament; the local
  SDK now reveals showdown hands only (`pokerbot_sdk.py`), so sims match reality.
- mw targets us: 3-bets 34.7% of our opens vs 20.0% of others' (round 2: 29.8 vs 25.9). We folded
  1122 times to mw 3-bets in 600 games (-2367 bb); mw folds 75-80% to 4-bets when deep.
- Models refit on round 3 (`data/r3_models2.json`), `R4ModelBot` with measured "3-bets vs our opens"
  multipliers (`R4_TARGET_MULT`), bench field `v4` = the 7 other finalists, one each.
- New: `vu_faced/vu_3b` counters (3-bets vs our opens, from history) and `P_VSUS_ON` (X4/D3 by frequency,
  no hole cards). Measured per firing vs mw in sim: light 4-bet jam +2.33 bb (fold = -2.05), D3 call -7.17 bb
  (bad: our OOP postflop in 3-bet pots) -> use X4 only (`P_D3_ON=0`, gate 0.26).
- Bug found: W3 4-bet-jam reward ignored villain's call (required equity 65% instead of ~43%) -> `P_W3_FIX`.
- A/B v4 (4000): X4-only -0.01, W3 fix -0.03, both +0.06 (all noise; the spot is rare, ~0.3/table).
- Points gap vs mw is mostly 2nd places (r3 sample: us 28/8/8 %, mw 25/16/8 % for 1st/2nd/3rd): we bust
  more before the final two. Re-jams (XR): +0.70 bb each overall but negative vs mw (-0.55), Daksh (-0.06),
  Vaibhaav (-3.16); top bust source.
- Sweep on v4 (3000 each): XR_N=3 -0.13 (worse), XR_MAX_BB=20 +0.10, RS_SIZE=2.8 +0.07 (4 runs, pooled ~+0.06),
  OPEN_BASE=0.48 +0.04, ICM 1.2 -0.06 / 0.85 +0.03, PUSH_MULT 0.9 +0.02, XR_PRIOR 0.6 -0.02 (log data/r4sweep.log).
- Timeouts in r3 cost ~nothing (31 in 600 games, mostly preflop folds / host stalls); not changed.
- ROUND-4 SUBMISSION: `contest/round4_submission_bot.py` = round-3 config - D3 + P_VSUS_ON (gate 0.26) + P_W3_FIX
  + P_RS_SIZE 2.8 + P_XR_MAX_BB 20 (cleaned build, 3897 lines). Paired vs round-3 config: v4 +0.03 +- 0.05,
  v3 +0.17 +- 0.06, light 3-bettors +0.63 +- 0.08, light 4-bettors +0.12 +- 0.07. Checks: adapter tests exact,
  build_check 0 exceptions / avg 0.053 ms / max 0.297 ms, replay_feed on real r3 hands: 0 exceptions, 97.8%
  preflop agreement with round 3 (differences = new 4-bets vs mw, XR cap, re-steal size).
- Bracket (rules text + entrant counts): field cut to half (rounded up) each round, apparently never below 8
  (33 -> 17 -> 9 -> 8). Remaining rounds 4 and 5 most likely with all 8; ranking = round reached, then average
  points in it -> only the last round decides; top 3 of the final win.
- mw's targeting is learned in-table (3-bet vs our opens / vs others: hands 0-24 9.6/7.9%, 25-74 26.5/17.2%,
  75-124 54.8/30.4%, 125+ 68.6/39.9%): it punishes over-folders at the table, it does not identify us.
- Open-size tell checked: controlled for position, our big opens get 3-bet MORE (late: 48% vs 23%), so not
  exploited; unified 2.05 bb opens A/B -0.04 +- 0.08 -> unchanged.
- Extraction gaps vs mw (bb/100, `tools/extract_gap.py`): Daksh (we -0.02, mw +1.91; we lose in Daksh-opened
  pots that go postflop although Daksh folds 72% to bets), Lorenzo (+4.55 vs +6.60, mostly our opens being
  3-bet). We out-extract mw vs Kashyap (+9.18 vs +5.13), Henrik, Shresth.

## 11. Opponent fingerprinting and new exploits (final prep)
- Kashyap times out on ~30% of its flop decisions (forced fold / check), 11-20% on turn/river, ~0 preflop.
- Fingerprinting (`bots/fp_core.py`, data `bots/fingerprints.py` generated by `tools/build_profiles.py`):
  naive Bayes over bet-size signatures (open size, unopened first action, 3-bet multiplier, postflop bet/pot)
  of the 7 other finalists + an "unknown" class. Held-out real r3 games: identified at 80-87% of tables,
  median hand 9 (Kashyap) to 44 (mw), ~1% wrong among finalists; an unknown bot (krishnah37) is misread
  ~55-70% of the time (similar sizes) -> profiles are injected as pseudo-counts (P_FP_W = 12) that real
  observations override. "One seat per bot" exclusion was tested and dropped (copies in r3 data broke it).
- On identification the bot's measured profile (from real r3 hands run through our own core, so in the exact
  counter definitions) is injected: fold-to-steal, 3-bet freq, fold-to-3-bet, 4-bet freq, call-vs-jam,
  fold-to-bet, 3-bet vs our opens, fold-to-4-bet, check-fold. E.g. Daksh check-fold 0.97, Henrik 0.92.
- A/B (P_FP_ON, on top of the round-4 package): v4 +0.238 +- 0.073, v3 (older versions) +0.134 +- 0.065.
- P_CHK_ON (small bet when villain checked and his check-fold rate >= 0.80): +0.05 +- 0.05 (v4).
- P_OJ_ON (open-jam instead of min-raise at <= 20 bb vs frequent re-raisers): +0.03 +- 0.05 (v4).
- Postflop betting variants (STAB min n / margin, c-bet gate): 0.00 / -0.03 / -0.11 -> unchanged.
- Combined P_CHK_ON + P_OJ_ON on fresh seeds: v4 -0.066 +- 0.040, v3 -0.029 +- 0.041 -> both OFF.
- ROUND-4 SUBMISSION (updated): `contest/round4_submission_bot.py` = round-4 package + P_FP_ON (4150 lines).
  Checks: adapter tests exact; build_check 0 exceptions, avg 0.054 ms, max 0.324 ms; real r3 hands (platform
  format, showdown-only): 0 exceptions, 174 correct identifications, 0 wrong, 97.8% preflop agreement.
- BEFORE THE FINAL: rebuild `bots/fingerprints.py` from the round-4 replays (the versions that will play):
  `python tools/build_profiles.py data/r4` for the identification test, then the profile block in this
  section's command history (build_signatures + build_profiles with the round-4 build), rebuild, re-check.

## 12. Round 4 result (2026-11-19)
- Round 4 (2500 tables, all 8 finalists): **1st, 3.08 pts/table** (mw 2.86, Daksh 2.48, Vaibhaav 1.84,
  Henrik 1.76, Lorenzo 1.75, Kashyap 1.68, Shresth 1.55). Wins: us 632 (25.3%), mw 554, Kashyap 258, Daksh 417.
  All 8 go through to the final. Faults: us 209 (0.08/table, r3 rate 0.05), Kashyap 5765.
- Bot played: `contest/round4_submission_bot.py` (round-4 package + P_FP_ON fingerprinting).
- Next: round-4 replays -> rebuild fingerprints/profiles (section 11), bet-size-only identification
  hardening (limps/folds alone must not identify Kashyap), re-validate, submit for the final.
- Round-4 replays: `data/r4/` (1200 games). Analysis: `tools/_r4_analyze.py`, `_r4_busts.py`, `_r4_targeting.py`,
  `_r4_xr.py`, `_r4_stages.py` (stage reach/conversion). Kashyap changed: folds 47% to re-jams (r3 72%), 29% to 3-bets.
- r3 fingerprints on r4 games: 42 wrong ids / 300 games; r4-trained (held out): 13. `tools/fp_idtest.py`.
- `bots/fp_core.py`: FP_MIN_SIZED = 2 (needs 2 bet-size observations; limps alone never identify).
- New counters `faced_xr/folded_xr` (opener vs all-in 3-bet, 8-25 bb) + `P_XRF_ON` (re-jam fold estimate from them),
  also in the profiles. `bots/fingerprints.py` rebuilt from r4 with `tools/rebuild_fp.py data/r4 data/prof_build.py`
  (prof_build = round-4 config with P_FP_ON=0 so injected priors don't leak into the measured profiles).
  r3 version kept at `data/fingerprints_r3.py`.
- Sim field `v5` = R5ModelBot (data/r4_models2.json, R5_TARGET_MULT); calibration corr 0.78 (`tools/calib_v5.py`).
  bench accepts built files as heroes: `file:path.py`.
- A/B (data/r5ab.log): new fp/profiles v5 -0.03 +- 0.07, v4 -0.05 +- 0.07; XRF v5 -0.01 +- 0.05, v4 -0.06 +- 0.05.
- Fresh seed 23 (4000, v5): new fp/profiles -0.01 +- 0.06; P_CHK_ON on top +0.02 +- 0.04. Pooled over 3 runs the
  refresh is ~-0.03 +- 0.04: no measurable gain. Candidates: data/cand_b1.py (refresh), b2 (+XRF), b4 (+CHK).
  Recommendation: final = contest/round4_submission_bot.py unchanged unless new evidence appears.
- FINAL CANDIDATE `contest/round5_candidate_bot.py` = round-4 config + r4 fingerprints/profiles (1200 games) +
  FP_MIN_SIZED=2 (XRF/CHK off). Checks: adapter variants exact (fidelity to_call mismatches as before, adapter
  unchanged); build_check 0 exceptions, avg 0.054 ms, max 0.267 ms; replay_feed 300 r4 games: 0 exceptions, 99.25%
  preflop agreement; identification on r4 games 900-1199 (in-sample): 1590 correct / 11 wrong vs round-4 file
  1544 / 22; house check 5.76 vs 5.75 (house tight/random bots still read as Kashyap: irrelevant in the final).
- Gemini review (2026-11-19): verified real bugs: STAB fires OOP after villain called (P_STABFIX_ON), 3-handed /
  HU-cap bust paid 0 in ICM (P_ICMFIX_ON, poker_engine/icm.py bust_fix), push/fold shove_bf reward = blinds only
  (BF 3.5 vs 1.07; P_PUSHFIX_ON), re-jam call range ignores opening range (P_XROF_ON), double-paired-board
  classifier (flush/straight -> air, quads -> 3; fixed in source), hand 0 not recorded (fixed), fp "unknown" class
  flat on UNOP (now population average; stale r3 sigs on r4: wrong ids 33 -> 12).
- A/B (data/r5fix.log, 4000 each): v5: ICM -0.02, PUSH -0.05, STAB +0.04, XROF +0.05; v4: ICM +0.04, PUSH -0.01,
  source fixes -0.05, STAB -0.05, XROF -0.03 (all +- 0.04-0.07). ALL combined (data/cand_all.py): v5 +0.04 +- 0.07,
  v4 -0.20 +- 0.07. No single culprit -> not adopted. NOTE: the sim is not deterministic (same config + seed gives
  A = 4.63..4.79 on v4), so real noise is larger than the printed SE.
- Sources now contain the source-level fixes (classifier, hand 0, fp unknown); round-4 sources backed up in
  data/*_r4sub.py. Final candidate unchanged: contest/round5_candidate_bot.py.
- BET-SIZE TELLS (Gemini bug 5): `tools/build_tells.py data/r4` -> `bots/tells.py` (bettor board-percentile histograms
  per bot / street / bet-pot bucket, first bets heads-up). `P_TELL_ON`: facing a first bet heads-up, equity =
  (1-w) MC + w * share of the villain's (fp-identified, else population) bets of this size made with a weaker hand;
  w = 0.35 / 0.55 / 0.85 by street; replaces the hand-made size bands. Held-out river calibration: predicted
  .07/.30/.49/.70/.91 vs actual .10/.27/.43/.70/.93. Real-hand river replay estimate: +400..450 bb / 600 games.
  A/B: switch alone v5 +0.085 +- 0.059, v4 +0.127 +- 0.054; full build vs round5_candidate (fresh seed 61):
  v5 +0.157 +- 0.064, v4 +0.033 +- 0.068.
  Latency: lookup ~70 us cold-free, 180-230 us inside a decision (cache) -> worst case 0.42 ms local (was 0.25);
  time guard P_TELL_MAX_ELAPSED_NS = 220 us.
- FINAL SUBMISSION: `contest/round5_submission_bot.py` (= data/cand_tell.py): round5_candidate + source fixes
  (classifier, hand 0, fp unknown class) + P_TELL_ON. Checks: adapter variants exact; replay_feed 300 r4 games
  0 exceptions, 98.72% preflop agreement; build_check 0 exceptions, avg 0.056, p99.9 0.349, max 0.421 ms;
  house check 5.73 (round-4 file 5.67), house-bot misreads 453 (was 843).
- Red-team review (Gemini): assumes mw rewrites against us; no sign in r4 (mw opens 2.2 bb 85% / 80% with us / others
  in the BB; its extra 3-bets vs us are in-table learning). Open risk of the tells: static data, no in-table check.
- 2026-11-19: user says round 5 is the PENULTIMATE round (round 6 = final). Submitting round5_submission_bot.py as is;
  round 5 = real test of P_TELL_ON. After round 5: download replays, check tell calls/folds vs round 4, timeouts,
  sizing changes per bot; rebuild tells + fingerprints from r5; consider showdown-based tell eviction.
