# How we built the bot: process log

Deep-ML No-Limit Pokerbot Tournament (Nov 2026). 8-seat sit-and-go, 2000 chips, blinds rising every
25 hands, 200-hand cap settled by ICM, points 10/5/2, **1 ms think time per decision**, Python + numpy.
This file reconstructs the whole process, from the first prototype to the round-3 resubmission, plus
what worked, what didn't, and the method we ended up with. Technical state: `PROJECT_STATE.md`.

---

## Phase 0: Prototype (Gemini, before this project log)
- Gemini built **ApexBot**, a hand-written heuristic core:
  - preflop percentile ranges, Nash push/fold under ~12 bb, ICM bubble factors;
  - a Bayesian archetype model of opponents (nit / TAG / LAG / station / AFK);
  - a Monte Carlo postflop equity engine.
- It came with a homemade simulator, a set of imagined "LLM-style" rival bots, and a `PROJECT_STATE.md`
  handover.
- Weak points we found later:
  - benchmarks only against invented opponents;
  - no knowledge of the real SDK;
  - no latency budget;
  - documented commands that didn't exist.

## Phase 1: Review and hardening, before entering (Claude)
**Goal: be ready for a single 60-minute contest session with only 3 submissions.**
1. **Read the saved contest website.** We extracted the real starter bot, the trained example bot, and
   the API documentation from the bundled JS (`reference/starter_module.js`). That gave us the SDK
   surface (`Bot.act(state)`, `Action.*`, state fields) and the blind schedule.
2. **Built a local stand-in SDK** (`pokerbot_sdk.py`, including `play_local`), so the exact submission
   code path runs offline. Wrote an adapter (`bots/sdk_adapter.py`) that converts SDK state into the
   core's format:
   - it reads every field defensively and infers seat, button and blinds;
   - it auto-detects the raise-amount convention;
   - 11 adapter-variant tests check that it reconstructs the state exactly.
3. **Latency first.** Monte Carlo got a hard deadline, numpy was vectorised over all 1326 hand combos,
   and `gc.freeze()` was added. Target: worst case well under 1 ms. We met it (round 1: 0.18 ms average).
4. **Exploit the known starter bot.** ApexPlus recognises it by its bet sizes and computes an exact
   best response against it (+0.2 points against starter-like bots).
5. **Benchmark harness.** `bench.py` runs parallel tables over many fields; paired seeds came later.
6. **Single-file build.** `submission_builder.py` inlines every module into one notebook cell with tuned
   overrides baked in, because the contest takes one cell.
7. **Runbook.** `contest/RUNBOOK.md` gave a minute-by-minute plan, decision rules, a probe cell, and a
   fallback bot.

## Phase 2: Contest session (60 minutes, 3 submissions)
- **Probe cell first.** It printed the real `state`, which surfaced 2 adapter assumptions to fix: both
  blinds are logged as `"blind"`, and one pot didn't match the contributions. Fixed before submission 1.
- **Submission 1:** demo score 5.0. All four losses were AK or JJ all-ins we were favoured to win.
- **During the session:** pulled public replays of the top entrants' demo games and measured their
  styles. The biggest finding: they open 23–90% of hands and fold 77–92% to re-raises.
- **Submission 2:** added the **re-steal** (EV-gated light 3-bets against wide openers who over-fold).
  Demo score 4.25 (demo tables are noise), but it was +0.24 to +0.39 on realistic fields. The user picked
  it as the entry.
- **Lesson:** the demo score against house bots doesn't matter. We didn't spend the 3rd submission on it.

## Phase 3: Between round 1 and round 2
Round 1: 3rd of 17 (3.83 points per table).
1. **Replays as data.** We downloaded about 560 round-1 games with all hole cards, gently: 3 parallel
   requests with delays.
2. **Measured, didn't guess.** We wrote analysis scripts for:
   - each bot's style;
   - our chip flow by hand type;
   - each bot's range by action, from hole cards;
   - fold rate against bet size;
   - responses to shoves and re-shoves;
   - stack-pressure behaviour;
   - bet-size tells.
3. **Opponent models from data.** Two versions:
   - v1: preflop tables by situation, stack depth and hand-strength decile;
   - v2: also postflop by hand-strength percentile, draw, aggressor, multiway and bet size; preflop by
     position.
   We calibrated against reality: our round-1 bot among the models should score near its real score, and
   model rankings should correlate with real rankings. Key insight: **sample actions by the hand's rank
   within its strength band (threshold-style), not independently**. That moved correlation from 0.66 to
   0.92.
4. **Exploits from the data, each as an EV-gated, per-opponent-learning switch:**
   - `P_W3`: defend against frequent 3-bettors, estimating their range from their observed frequency;
   - `P_XJ`: open-shove at ≤20 bb against players who rarely call shoves; call rates are learned from
     the end-of-hand chip deltas;
   - `P_STAB`: half-pot bet when checked to, against observed over-folders. 0.5× pot, not 0.33×,
     because real bots defend small bets;
   - `P_XR`: re-shove over a single open at 8–25 bb. Openers folded 87% in the data; we had re-shoved 2%
     of the time.
5. **Validation with paired A/B** (same seeds, 1,500–4,000 tables) on several fields, adopting only what
   was positive on the realistic fields and not clearly negative elsewhere.
6. **External research.** Two literature reports. We used them as leads and checked each against our
   data; most ideas were already covered or didn't fit 1 ms.
7. **Probe of the end-of-hand result:** it contains chip deltas, the full action list, and **every
   player's hole cards**. One fix followed: our re-shove observations were being counted twice.

## Phase 4: Between round 2 and round 3
Round 2: **1st of 17** (4.23; mw 4.21). Round 3 field: 9 bots.
1. **600 round-2 replays.** Leaks found:
   - against mw we lost −438 bb, mostly by folding to mw's 3-bets, even with top-10% hands when deep;
   - our biggest bust source is our own re-shoves (but they're +1,822 bb overall).
2. **Hole-card features** (shown 3-bet ranges per opponent, fold-to-4-bet): light 4-bets (`P_X4`) and a
   deep 3-bet defence (`P_D3`). Neutral in simulation, so not adopted at first.
3. **Model bugs found by checking the simulation against the data:**
   - depth buckets mixed 15–40 bb;
   - facing a 3-bet and facing a 4-bet shared one context;
   - too much backoff to the population;
   - the threshold noise was a weighted sum, which isn't uniform and under-selected rare actions.
   After the fixes, simulated mw reproduced the real 3-bet rate per situation (34/23/16/8% vs
   33/25/17/8%).
4. **Parameter sweep** (16 variants): all within noise, so the bot sits at a local optimum for this field.
5. **Adversarial thinking: "what would they do to beat us?"** We built counter-strategy bots on top of
   the models:
   - light 3-bettor;
   - light 4-bettor;
   - re-shover;
   - wide caller of shoves;
   - postflop raiser.

   What they revealed:
   - Light 3-bettors cost us about 0.3 points per table. The deep 3-bet defence plus light 4-bets fix
     that (+0.45 to +0.63).
   - A hardcoded hole: facing a 4-bet, we folded everything below the top 4% of hands. The new
     **adaptive 4-bet defence** (`P_W4`) fixes it (+0.08 to +0.12 against light 4-bettors).
   - Wide callers and re-shovers barely hurt us; the per-opponent learning already adapts.
   - Learning bluff rates from hole cards (`P_HB`) didn't help, because there are too few observations
     per table. Left off.
6. **Gating.** The ungated 3-bet defence cost −0.13 on the normal field in one run. Gated to clearly
   light 3-bettors only, it's about neutral there (−0.03 / 0.00) and keeps the protection.
7. **Pipeline checks before building:**
   - a shadow bot fed the **exact platform result format** learns identical counters;
   - a smoke test finds no exceptions (the adapter would otherwise silently check/fold);
   - unloaded think time averages 0.056 ms (worst 0.295 ms).
8. **Round-3 file:** `contest/round3_submission_bot.py`.

## Phase 5: After the round-3 submission (cleanup, external code review)
- An external review claimed four bugs. We checked each against the code: one was intended behaviour
  (polarised 4-bets), one was wrong (heads-up seat order is correct for postflop), two were true but had no
  effect. The useful insight: the starter-bot layer was dead code for this field (0 of 814 decisions).
- We **didn't rewrite from scratch**. That would discard all validation, and the adapter turns bugs into
  silent folds. We cleaned up in the sources instead, with a **decision-equivalence harness**: record real
  game events, replay them into the old and new build with a seeded random generator and no time limits,
  and require identical decisions (0 of 4,950 differed). The builder now strips unreferenced code
  automatically. The build went from 4,680 to 3,847 lines.
- The **exact river equity** suggestion was implemented and verified (it matches a 40,000-sample Monte Carlo)
  and gained about +0.04. It stays off because it pushes worst-case think time too close to the platform's
  1 ms limit. Lesson: check the platform's tail latency, not just the local average.

## Phase 6: After round 3 (2nd behind mw) to the final
- **Validate against reality, not only against the simulation.** A new tool feeds real tournament hands,
  rebuilt in the platform's exact formats, into our submitted bot and compares its decisions with what it
  really did. That showed the tournament reveals **only showdown hands**, not every hole card as the
  `play_local` probe had: preflop agreement was 99.55% with showdown-only information and 98.42% with all
  holes. So the round-3 defences that relied on hole cards never fired, and our simulator had been giving
  the bot information it never gets. The simulator now reveals showdown hands only.
- **mw had adapted to us.** It 3-bet 35% of our opens (20% of other players') and folded 75–80% to
  4-bets. We rebuilt the light 4-bet on frequency counted from the action history. Measuring each firing in
  the simulation showed that calling mw's 3-bets lost money while shoving over them won. A pot-odds bug
  that had disabled the old 4-bet-shove defence was fixed.
- **Final field model:** all 8 finalists at every table, fitted on round 3, with each bot's measured
  "3-bets against us" multiplier.
- **Points analysis:** the gap to mw came mostly from second places (we bust before the final two more
  often), so the re-shove was capped at 20 bb.
- **Opponent fingerprinting (the biggest gain of the late phase, +0.24 in sim):** with the same 8 bots at
  every table, the bot recognises each opponent from its near-deterministic bet sizes (median hand 9–44)
  and loads that bot's real statistics, measured by running the replays through our own core, as
  pseudo-observations. Every existing exploit then starts from the right numbers instead of learning for
  dozens of hands. Validated on held-out real hands: about 1% misidentification among the finalists.
- **Ideas that didn't survive fresh seeds** (check-fold bets, open-shoving into re-raisers, betting
  variants): each looked slightly positive once, then negative in combination. Not adopted.
- **Round-4 file:** `contest/round4_submission_bot.py`.

---

## The method we ended up with
1. **Measure before changing.** Every idea starts as a question to the replays: how often, against
   whom, for how many bb.
2. **EV-gated exploits that learn per opponent.** Each exploit computes its own EV from that opponent's
   observed rates, with a cautious prior, so it switches off against players who don't have the leak.
   This is also our main defence against opponents who adapt.
3. **Calibrated opponent models.** A test field is only useful if its scores look like reality. Always
   check the simulation's situation-by-situation frequencies against the replays; that is how the
   biggest model bugs were found.
4. **Paired A/B with enough tables.** 2,500–5,000 tables per field. Re-run promising results on fresh
   seeds: a +0.17 at two standard errors more than once turned into −0.02.
5. **Several fields, including adversarial ones.** "Neutral now, strongly positive against plausible
   counter-play" is worth adopting as insurance. "Strongly positive against a crude model" is not enough
   on its own.
6. **Respect the 1 ms limit.** Hard deadlines inside Monte Carlo, vectorised numpy, `gc.freeze()`,
   latency checked on every build.
7. **Check your assumptions about the platform against real data.** Replay real hands in the platform's
   format and compare decisions (`tools/replay_feed.py`); one probe of `play_local` isn't the tournament.
8. **Prove refactors are behaviour-neutral.** Record/replay equivalence (`tools/equiv.py`) before any cleanup.
9. **Don't break a winning file.** Keep every round's submission (`contest/round*_submission_bot.py`),
   switch new features on through `bots/tuned_params.py`, run the adapter tests, a smoke test and a
   latency check before each build.
10. **Be gentle with external APIs.** Throttled replay downloads (3 parallel, with delays).

## What didn't work (and why)
- Tuning against the house/demo bots: it has no effect on the tournament.
- Small (0.33× pot) bets: real bots call them; 0.5× pot works.
- Neural, solver and FGS-ICM ideas from the literature: too slow or unprovable at 1 ms.
- Learning postflop bluff rates from hole cards: too few observations per table.
- Changes that only looked good against the old, too-weak models (e.g. the 4-bet exploit before the
  model fixes).

## Results so far
| Stage | Result |
|---|---|
| Demo, submission 1 | 5.00 points |
| Demo, submission 2 (re-steal, chosen) | 4.25 points |
| Round 1 | 3rd of 17, 3.83 points/table |
| Round 2 | **1st of 17, 4.23 points/table** (mw 4.21) |
| Round 3 | 2nd of 9, 2.97 points/table (mw 3.35) |
| Final (round 4) | submitted: `contest/round4_submission_bot.py` |
