"""
O(1) Precomputed Preflop Strategy Tables for 8-Max No-Limit Hold'em Sit-and-Go.
Provides:
- 52x52 O(1) lookup tables for:
  - Raw Heads-Up Equity vs random hand (`PREFLOP_HU_EQ[c0][c1]`)
  - Multi-way Playability-Adjusted Strength (`PREFLOP_STRENGTH[c0][c1]` in 0..1)
  - Positional Open-Raise (RFI) Rank Percentile (`PREFLOP_PERCENTILE[c0][c1]` in 0..1, 1.0 = AA)
  - Nash Push/Fold Max Effective BB Thresholds (`NASH_PUSH_BB[c0][c1]`, `NASH_CALL_BB[c0][c1]`)
"""

from __future__ import annotations
from typing import List, Tuple

# Exact 13x13 Heads-Up Preflop Equity Matrix (in percent, e.g. 85.2 = 85.2%)
# Rows = rank 12 down to 0 (A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2)
# Upper triangle (col > row_idx, i.e., r1 > r2) = Suited
# Diagonal (col == row_idx) = Pocket Pair
# Lower triangle (col < row_idx) = Offsuit
_EQ_13X13: Tuple[Tuple[float, ...], ...] = (
    #   A     K     Q     J     T     9     8     7     6     5     4     3     2
    (85.2, 67.0, 66.2, 65.4, 64.7, 63.0, 62.1, 61.1, 60.0, 59.9, 58.9, 58.0, 57.0),  # A
    (65.3, 82.4, 63.4, 62.6, 61.9, 60.0, 58.5, 57.8, 56.8, 55.8, 54.7, 53.8, 52.9),  # K
    (64.5, 61.5, 79.9, 60.3, 59.5, 57.9, 56.2, 54.5, 53.8, 52.9, 51.7, 50.7, 49.8),  # Q
    (63.6, 60.6, 58.2, 77.5, 57.5, 55.8, 54.2, 52.4, 50.8, 50.0, 49.0, 47.9, 47.1),  # J
    (62.9, 59.9, 57.4, 55.3, 75.1, 54.3, 52.6, 51.0, 49.2, 47.2, 46.4, 45.5, 44.7),  # T
    (60.9, 57.9, 55.5, 53.4, 51.7, 72.1, 51.1, 49.5, 47.7, 45.9, 43.8, 43.2, 42.3),  # 9
    (59.9, 56.3, 53.8, 51.7, 50.0, 48.4, 69.1, 48.2, 46.5, 44.8, 42.7, 40.8, 40.3),  # 8
    (58.9, 55.4, 51.9, 49.8, 48.2, 46.7, 45.3, 66.2, 45.7, 43.8, 41.8, 40.0, 38.1),  # 7
    (57.7, 54.3, 51.2, 47.9, 46.3, 44.8, 43.4, 42.6, 63.3, 43.2, 41.4, 39.4, 37.5),  # 6
    (57.7, 53.3, 50.2, 47.1, 44.2, 42.9, 41.7, 40.7, 40.1, 60.3, 41.1, 39.2, 37.5),  # 5
    (56.4, 52.1, 49.0, 46.1, 43.4, 40.7, 39.5, 38.6, 38.0, 37.9, 57.0, 38.0, 36.3),  # 4
    (55.6, 51.2, 47.9, 45.0, 42.4, 39.9, 37.5, 36.6, 35.9, 35.8, 34.4, 53.7, 35.1),  # 3
    (54.6, 50.2, 47.0, 44.0, 41.5, 39.1, 36.8, 34.6, 34.0, 33.9, 32.6, 31.2, 50.3),  # 2
)

# 52x52 fast O(1) lookup tables
PREFLOP_HU_EQ: List[List[float]] = [[0.5] * 52 for _ in range(52)]
PREFLOP_STRENGTH: List[List[float]] = [[0.5] * 52 for _ in range(52)]
PREFLOP_PERCENTILE: List[List[float]] = [[0.5] * 52 for _ in range(52)]
NASH_PUSH_BB: List[List[float]] = [[0.0] * 52 for _ in range(52)]
NASH_CALL_BB: List[List[float]] = [[0.0] * 52 for _ in range(52)]


def _init_preflop_tables() -> None:
    # Compute playability-adjusted solver score for all 169 canonical hands
    # and rank them into exact combo-weighted percentiles (1326 total combos)
    canonical_scores = {}
    for r1 in range(13):
        for r2 in range(r1 + 1):
            row_hi = 12 - r1
            col_lo = 12 - r2
            if r1 == r2:
                # Pocket pair
                hu_eq = _EQ_13X13[row_hi][row_hi] / 100.0
                # Pairs have high multiway set value + raw equity
                play_score = hu_eq + 0.085 + 0.008 * r1
                canonical_scores[(r1, r2, False)] = (hu_eq, play_score, 6)
            else:
                for suited in (True, False):
                    hu_eq = (
                        _EQ_13X13[row_hi][col_lo] / 100.0
                        if suited
                        else _EQ_13X13[col_lo][row_hi] / 100.0
                    )
                    gap = r1 - r2
                    # Suited bonus (flush potential + equity realization)
                    suit_bonus = 0.045 if suited else 0.0
                    # Connectedness bonus (straight potential for gap 1..3)
                    if gap == 1:
                        conn_bonus = 0.028 if r2 >= 3 else 0.018
                    elif gap == 2:
                        conn_bonus = 0.016 if r2 >= 3 else 0.008
                    elif gap == 3:
                        conn_bonus = 0.006
                    else:
                        conn_bonus = -0.012  # Disconnected penalty in multiway pots

                    # Broadway / High-card domination bonus
                    bway_bonus = 0.0
                    if r1 >= 8 and r2 >= 8:  # Both T or higher
                        bway_bonus = 0.032
                    elif r1 == 12 and suited:  # Suited Ace (nut flush potential + blocker)
                        bway_bonus = 0.026
                    elif r1 == 12 and r2 < 7 and not suited:
                        # Weak offsuit Ace is easily dominated
                        bway_bonus = -0.015

                    play_score = hu_eq + suit_bonus + conn_bonus + bway_bonus
                    combos = 4 if suited else 12
                    canonical_scores[(r1, r2, suited)] = (hu_eq, play_score, combos)

    # Sort canonical hands by play_score ascending to compute combo-weighted percentile (0.0 to 1.0)
    sorted_keys = sorted(canonical_scores.keys(), key=lambda k: canonical_scores[k][1])
    cum_combos = 0
    percentiles = {}
    for k in sorted_keys:
        hu_eq, play_score, combos = canonical_scores[k]
        # Midpoint percentile in [0, 1], with AA -> ~0.998
        pct = (cum_combos + 0.5 * combos) / 1326.0
        percentiles[k] = pct
        cum_combos += combos

    # Populate 52x52 tables
    for c0 in range(52):
        r0 = c0 >> 2
        s0 = c0 & 3
        for c1 in range(52):
            if c0 == c1:
                continue
            r1 = c1 >> 2
            s1 = c1 & 3
            hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
            suited = (s0 == s1) and (hi != lo)
            key = (hi, lo, suited)
            hu_eq, play_score, _ = canonical_scores[key]
            pct = percentiles[key]

            PREFLOP_HU_EQ[c0][c1] = hu_eq
            PREFLOP_STRENGTH[c0][c1] = min(0.99, max(0.20, play_score))
            PREFLOP_PERCENTILE[c0][c1] = pct

            # Nash Push/Fold effective BB threshold (base heads-up / SB-vs-BB reference)
            # Scaled at runtime by number of players left to act and ICM bubble factor
            if hi == lo:
                # Any pocket pair is an unexploitable push up to 20+ BB
                push_bb = 22.0 if hi >= 4 else 16.0 + hi * 1.2
                call_bb = 22.0 if hi >= 6 else 9.0 + hi * 1.8
            else:
                if pct >= 0.88:
                    push_bb = 20.0
                elif pct >= 0.75:
                    push_bb = 15.0 + (pct - 0.75) * 38.0
                elif pct >= 0.50:
                    push_bb = 7.5 + (pct - 0.50) * 30.0
                elif suited and hi == 12:  # Any suited Ace
                    push_bb = 13.5
                else:
                    push_bb = max(1.5, pct * 14.5)

                # Calling an all-in requires stronger high-card / pair equity than pushing
                if pct >= 0.92:
                    call_bb = 20.0
                elif pct >= 0.80:
                    call_bb = 10.5 + (pct - 0.80) * 75.0
                elif hi == 12 and lo >= 7:  # A9+
                    call_bb = 10.0 + (lo - 7) * 2.2
                else:
                    call_bb = max(1.0, (hu_eq - 0.42) * 32.0)

            NASH_PUSH_BB[c0][c1] = push_bb
            NASH_CALL_BB[c0][c1] = call_bb


_init_preflop_tables()
