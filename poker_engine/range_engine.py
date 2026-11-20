"""
Vectorized range engine: hand scores for all 1326 hole-card combos on a board in one
numpy pass (~tens of microseconds), plus per-combo preflop rank tables.

Used to model opponents whose policy is a function of their made hand (the contest's
starter bot and its edits): an observed action restricts their range to the combos
whose strength is consistent with it.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from poker_engine.evaluator import (
    CARD_PRIME,
    CARD_SUIT_BIT,
    CARD_SUIT_WEIGHT,
    FLUSH_SCORE,
    FLUSH_SUIT_SHIFT,
    PRIME_TO_SCORE,
)
from poker_engine.preflop_multiway import MULTIWAY_EQ

# All 1326 combos (c0 < c1)
COMBO_C0, COMBO_C1 = (a.astype(np.int64) for a in np.triu_indices(52, 1))
N_COMBOS = COMBO_C0.shape[0]

_P = np.array(CARD_PRIME, dtype=np.int64)
_SW = np.array(CARD_SUIT_WEIGHT, dtype=np.int64)
_SB = np.array(CARD_SUIT_BIT, dtype=np.int64)
_FSS = np.array(FLUSH_SUIT_SHIFT, dtype=np.int64)
_FSC = np.array(FLUSH_SCORE, dtype=np.int64)
_KEYS = np.array(sorted(PRIME_TO_SCORE), dtype=np.int64)
_VALS = np.array([PRIME_TO_SCORE[k] for k in sorted(PRIME_TO_SCORE)], dtype=np.int64)
_NKEYS = _KEYS.shape[0]

_HP = _P[COMBO_C0] * _P[COMBO_C1]
_HSW = _SW[COMBO_C0] + _SW[COMBO_C1]
_HSB = _SB[COMBO_C0] | _SB[COMBO_C1]
COMBO_MASK = (np.int64(1) << COMBO_C0) | (np.int64(1) << COMBO_C1)

# combo index lookup: COMBO_INDEX[c0][c1]
COMBO_INDEX = np.full((52, 52), -1, dtype=np.int64)
COMBO_INDEX[COMBO_C0, COMBO_C1] = np.arange(N_COMBOS)
COMBO_INDEX[COMBO_C1, COMBO_C0] = np.arange(N_COMBOS)

# Preflop rank of each combo vs n opponents (n = 1..7), 0 = worst class .. 1 = AA
_r0 = COMBO_C0 >> 2
_r1 = COMBO_C1 >> 2
_hi = np.maximum(_r0, _r1)
_lo = np.minimum(_r0, _r1)
_suited = ((COMBO_C0 & 3) == (COMBO_C1 & 3)) & (_hi != _lo)
PREFLOP_RANK = np.zeros((8, N_COMBOS), dtype=np.float64)
PREFLOP_EQ = np.zeros((8, N_COMBOS), dtype=np.float64)
for _n in range(1, 8):
    _order = sorted(MULTIWAY_EQ, key=lambda k: MULTIWAY_EQ[k][_n - 1])
    _rank = {k: i / (len(_order) - 1) for i, k in enumerate(_order)}
    for _i in range(N_COMBOS):
        _k = (int(_hi[_i]), int(_lo[_i]), bool(_suited[_i]))
        PREFLOP_RANK[_n, _i] = _rank[_k]
        PREFLOP_EQ[_n, _i] = MULTIWAY_EQ[_k][_n - 1]
PREFLOP_RANK[0] = PREFLOP_RANK[1]
PREFLOP_EQ[0] = PREFLOP_EQ[1]


def board_info(board: Sequence[int]) -> Tuple[int, int, int, int]:
    bp, bsw, bsb, bm = 1, 0, 0, 0
    for c in board:
        bp *= CARD_PRIME[c]
        bsw += CARD_SUIT_WEIGHT[c]
        bsb |= CARD_SUIT_BIT[c]
        bm |= 1 << c
    return bp, bsw, bsb, bm


def combo_scores(board: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """(score per combo, combo-does-not-collide-with-board mask) for a 3..5 card board."""
    bp, bsw, bsb, bm = board_info(board)
    prod = _HP * bp
    idx = np.searchsorted(_KEYS, prod)
    np.minimum(idx, _NKEYS - 1, out=idx)
    score = _VALS[idx]
    sh = _FSS[_HSW + bsw]
    fl = sh >= 0
    if fl.any():
        bits = ((_HSB | bsb) >> np.where(fl, sh, 0)) & 0x1FFF
        score = np.where(fl, _FSC[bits], score)
    valid = (COMBO_MASK & bm) == 0
    return score, valid


class BoardCache:
    """Caches combo scores per board within a hand."""

    def __init__(self) -> None:
        self._c: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, board: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        key = tuple(board)
        r = self._c.get(key)
        if r is None:
            if len(self._c) > 64:
                self._c.clear()
            r = combo_scores(board)
            self._c[key] = r
        return r
