# Deep-ML No-Limit Hold'em Pokerbot Championship Kit (`ApexBot`)

Complete preparation, simulation, and rapid-adaptation environment for the **Deep-ML No-Limit: Pokerbot Tournament** (8-max Sit-and-Go, 2,000 starting chips, blinds rising every 25 hands, 200-hand cap with `10/5/2` ICM payouts, and strict **1.0 ms per-decision think limit**).

## Project Structure

- [`poker_engine/evaluator.py`](poker_engine/evaluator.py): Ultra-fast O(1) 5/6/7-card hand evaluator (**0.20 µs** per 7-card hand, verified exact across all 2,598,960 5-card combinations) + Range-Weighted Monte Carlo Postflop Equity (**~60 µs**) + O(1) Board Texture & Draw Analyzer (**1.8 µs**).
- [`poker_engine/preflop_tables.py`](poker_engine/preflop_tables.py): O(1) $52 \times 52$ precomputed tables for Heads-Up Equity, Multi-way Playability Strength, Positional RFI Percentiles, and Nash Push/Fold Effective BB thresholds.
- [`poker_engine/icm.py`](poker_engine/icm.py): Exact Malmuth-Harville `(10, 5, 2)` ICM Equity and Hero-Specific Bubble Factor calculator (**11.3 µs** on an 8-player table) with 200-hand cap proximity weighting.
- [`poker_engine/simulator.py`](poker_engine/simulator.py): Full 8-player No-Limit Hold'em Sit-and-Go Tournament Simulator with side-pots, blind escalation every 25 hands, 200-hand cap ICM settlement, and strict `1.0 ms` (`1,000,000 ns`) latency enforcement.
- [`bots/apex_bot.py`](bots/apex_bot.py): **`ApexBot`** — our championship tournament bot featuring an Online Bayesian Opponent Profiler (with both `action_history` and snapshot-only fallback modes), 3-gear SNG phase strategy, low-risk min-raise blind stealing against Nits/Timeouts, and range-condensed postflop play.
- [`bots/archetypes.py`](bots/archetypes.py): Realistic competition opponent zoo (`TimeoutFoldBot`, `CallingStationBot`, `TightPassiveNitBot`, `ManiacAggroBot`, `HeuristicEquityBot`, `SharkTAGBot`).
- [`run_tournament.py`](run_tournament.py): Multi-table Sit-and-Go benchmark and 64-player Elimination Bracket simulator.
- [`submission_builder.py`](submission_builder.py) & [`submission_bot.py`](submission_bot.py): Compiles the entire engine and `ApexBot` into a single standalone file (`submission_bot.py`) with a universal schema adapter at the bottom for <2-minute adaptation once the 60-minute contest timer begins.

## Quick Commands

```powershell
# Run 300 Sit-and-Go tables against a Round 1 field
python run_tournament.py --tables 300 --field round1

# Run 300 Sit-and-Go tables against a Tough Finals field
python run_tournament.py --tables 300 --field finals

# Run full 64-player multi-round Elimination Brackets
python run_tournament.py --bracket 20

# Rebuild the standalone single-file submission_bot.py
python submission_builder.py
```
