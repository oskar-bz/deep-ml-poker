# Strategy overrides baked into submission_bot.py by submission_builder.py.
# P_RESTEAL_ON: re-steal vs wide openers that over-fold to 3-bets (real entrant replays:
# open 35-90%, fold 77-92% to re-raises). A/B, 1500-2000 tables/field: replay-model +0.24,
# llm +0.39, entrants2 +0.06, realistic -0.08, starters -0.06, mirror -0.11 (last three noise).
# P_W3_ON (round 2): call jams / 4-bet jam vs empirically wide 3-bettors (round-1 replays: mw
# re-jams 18% of opens with median hand top-26%, we folded 92% to him). Paired 2000 tables:
# r1mw +0.14, llm +0.16, entrants2 +0.06, r1top +0.02, realistic +0.00.
# P_XJ_ON (round 2): EV open-jam <= 20 bb vs players who rarely call jams (round-1: most bots call
# open-jams 3-15% even 3-handed / heads-up). Per-seat call rate learned from shown cards after our
# jams. Paired 2000 tables (refit models): r1mid +0.25, llm +0.35, r1top +0.06, r1mw -0.01,
# realistic -0.02. 25 bb depth tested: +0.05..+0.15 on r1 fields but -0.05..-0.07 elsewhere -> kept 20.
# P_STAB_ON (round 2): EV half-pot stab when checked to heads-up vs players whose observed
# fold-to-bet is high (round-1 real fold rate to 0.4-0.7 pot bets ~52%, to <0.4 pot only ~42%,
# so 0.5 pot, not 0.33). Paired 1500 tables, size-aware models: r1top +0.42, r1mid +0.30,
# r1mw +0.31, llm +0.47, mirror +0.15, realistic/starters/entrants2 ~0. 0.33 pot: r1mid -0.57.
# P_XR_ON (round 2, last): EV re-jam over a single open at 8-25 bb (round-1: openers fold 87% to
# re-jams at 12-25 bb; we re-jammed 2%). Opener fold rate learned from deltas. v2 models, paired 2500:
# v2final +0.63, v2r1 +0.27, v2mirror +0.24; llm +0.34, realistic/entrants2 ~0, starters -0.17.
# Round 3 (defence package, hole cards from on_hand_end): P_D3_ON+P_D3_GATE (call/4-bet deep 3-bets only vs
# clearly light 3-bettors), P_X4_ON (4-bet light 3-bettors shown by hole cards), P_W4_ON (adaptive 4-bet
# defence). Paired 3000 tables: v3 -0.03 / 0.00, adversary light 3-bettors +0.45, light 4-bettors +0.08,
# re-jammers +0.03, mirror +0.10.
# Round 4 prep: P_RIVX_ON exact heads-up river profile (same model, no sampling noise). Paired 4000 tables,
# 5 runs: v3 +0.07/+0.05, v3mirror +0.06, v2final 0.00, v3adv3 +0.01 (pooled ~+0.04 +- 0.03).
# NOT enabled: it raises our local worst-case think time 0.31 -> 0.47 ms; the platform stretches tails
# 1.4-2.1x (round-2 per-game max: median 0.43, p95 0.65, p99 1.42 ms) -> too close to the 1 ms limit.
# Round-4 package vs round-3 config (paired): v4 +0.03 (6000), v3 +0.17 (4000), light 3-bettors +0.63,
# light 4-bettors +0.12 (3000 each), with showdown-only information as in the real tournament.
TUNED = {"P_RESTEAL_ON": 1.0, "P_W3_ON": 1.0, "P_XJ_ON": 1.0, "P_STAB_ON": 1.0, "P_STAB_FRAC": 0.5,
         "P_XR_ON": 1.0, "P_XR_MARGIN": 1.5, "P_XR_PRIOR": 0.70,
         "P_D3_GATE": 1.0, "P_X4_ON": 1.0, "P_W4_ON": 1.0,
         # round 4 (final): the tournament reveals only showdown hands, so light 3-bettors are found by
         # frequency vs our opens (P_VSUS_*); D3 calls removed (lost ~5 bb vs folding in 3-bet pots OOP);
         # W3 jam reward fix; smaller re-steal; re-jam only up to 20 bb.
         "P_VSUS_ON": 1.0, "P_VSUS_GATE": 0.26, "P_W3_FIX": 1.0, "P_RS_SIZE": 2.8, "P_XR_MAX_BB": 20.0,
         # opponent fingerprinting (bots/fp_core.py + bots/fingerprints.py from round-3 replays):
         # v4 +0.24 +- 0.07, older versions (v3) +0.13 +- 0.07. P_CHK_ON / P_OJ_ON tested: combined -0.07 -> off.
         "P_FP_ON": 1.0}
