"""
Benchmark a hero bot through the real SDK code path against realistic fields.

    python bench.py --hero apex_sdk --field sdkmix --tables 400
    python bench.py --hero apex_sdk --field all --tables 300      # every field

Heroes / opponents are referenced by registry name so tables can run in worker
processes. Reports average points (10/5/2 + ICM at cap) with standard error;
random baseline is 17/8 = 2.125.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Tuple


def _registry() -> Dict[str, Callable[[], object]]:
    from pokerbot_sdk import SDKBotDriver
    from bots import sdk_field as F
    from bots.apex_bot import ApexBot
    from bots.apex_plus import ApexPlus
    from bots.sdk_adapter import MyBot
    from bots.archetypes import (CallingStationBot, HeuristicEquityBot, LLM_CFR_GTO_Bot,
                                 LLM_ICM_ExploiterBot, LLM_TrapCounterBot, ManiacAggroBot,
                                 SharkTAGBot, TightPassiveNitBot, TimeoutFoldBot)
    from bots.rival_neural_lag import RivalNeuralLAGBot
    from bots.rival_bayesian_profiler import RivalBayesianProfilerBot
    from bots.rival_claude_style import RivalClaudeStyleBot
    from bots.rival_icm_sniper import RivalICMSniperBot

    def sdk(factory, name, exempt=True):
        return lambda: SDKBotDriver(factory(), name, exempt_timing=exempt)

    reg: Dict[str, Callable[[], object]] = {
        # heroes (timed: they must respect 1 ms)
        "apex_raw": lambda: ApexBot(),
        "apex_sdk": lambda: SDKBotDriver(MyBot(ApexBot()), "apex_sdk"),
        "plus_raw": lambda: ApexPlus(),
        "plus_sdk": lambda: SDKBotDriver(MyBot(ApexPlus()), "plus_sdk"),
        # SDK-style opponents (reference implementations, not timed)
        "starter": sdk(F.StarterBot, "starter"),
        "starter2": sdk(F.StarterShortBot, "starter2"),
        "m_devin": sdk(lambda: F.EntrantModelBot(0.90, 0.92, 0.2, random.randrange(1 << 30)), "m_devin"),
        "m_henrik": sdk(lambda: F.EntrantModelBot(0.67, 0.89, 0.3, random.randrange(1 << 30)), "m_henrik"),
        "m_scott": sdk(lambda: F.EntrantModelBot(0.55, 0.84, 0.25, random.randrange(1 << 30)), "m_scott"),
        "m_lorenzo": sdk(lambda: F.EntrantModelBot(0.40, 0.84, 0.4, random.randrange(1 << 30)), "m_lorenzo"),
        "m_evan": sdk(lambda: F.EntrantModelBot(0.34, 0.78, 0.6, random.randrange(1 << 30)), "m_evan"),
        "m_vidvuds": sdk(lambda: F.EntrantModelBot(0.23, 0.77, 0.3, random.randrange(1 << 30)), "m_vidvuds"),
        "m_shresth": sdk(lambda: F.EntrantModelBot(0.76, 0.53, 0.3, random.randrange(1 << 30)), "m_shresth"),
        "lr30": sdk(lambda: F.LooseRaiserBot(0.7, 0.2, random.randrange(1 << 30)), "lr30"),
        "lr50": sdk(lambda: F.LooseRaiserBot(0.5, 0.3, random.randrange(1 << 30)), "lr50"),
        "lr70": sdk(lambda: F.LooseRaiserBot(0.3, 0.4, random.randrange(1 << 30)), "lr70"),
        "adv3b": sdk(lambda: F.Adv3BetBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json"), "adv3b"),
        "advcall": sdk(lambda: F.AdvCallerBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json"), "advcall"),
        "adv4b": sdk(lambda: F.Adv4BetBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json"), "adv4b"),
        "advraise": sdk(lambda: F.AdvRaiseBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json"), "advraise"),
        "advjam": sdk(lambda: F.AdvJamBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json"), "advjam"),
        "r4_mw": sdk(lambda: F.R4ModelBot("mw", random.randrange(1 << 30)), "r4_mw"),
        "r4_dak": sdk(lambda: F.R4ModelBot("Daksh Malhan", random.randrange(1 << 30)), "r4_dak"),
        "r4_vai": sdk(lambda: F.R4ModelBot("Vaibhaav Tiwari", random.randrange(1 << 30)), "r4_vai"),
        "r4_lor": sdk(lambda: F.R4ModelBot("Lorenzo P.", random.randrange(1 << 30)), "r4_lor"),
        "r4_hen": sdk(lambda: F.R4ModelBot("Henrik Gombos", random.randrange(1 << 30)), "r4_hen"),
        "r4_shr": sdk(lambda: F.R4ModelBot("Shresth Rana", random.randrange(1 << 30)), "r4_shr"),
        "r4_kas": sdk(lambda: F.R4ModelBot("Kashyap Patel", random.randrange(1 << 30)), "r4_kas"),
        "r5_mw": sdk(lambda: F.R5ModelBot("mw", random.randrange(1 << 30)), "r5_mw"),
        "r5_dak": sdk(lambda: F.R5ModelBot("Daksh Malhan", random.randrange(1 << 30)), "r5_dak"),
        "r5_vai": sdk(lambda: F.R5ModelBot("Vaibhaav Tiwari", random.randrange(1 << 30)), "r5_vai"),
        "r5_lor": sdk(lambda: F.R5ModelBot("Lorenzo P.", random.randrange(1 << 30)), "r5_lor"),
        "r5_hen": sdk(lambda: F.R5ModelBot("Henrik Gombos", random.randrange(1 << 30)), "r5_hen"),
        "r5_shr": sdk(lambda: F.R5ModelBot("Shresth Rana", random.randrange(1 << 30)), "r5_shr"),
        "r5_kas": sdk(lambda: F.R5ModelBot("Kashyap Patel", random.randrange(1 << 30)), "r5_kas"),
        "v3": sdk(lambda: F.R2ModelBot(random.choice(F.R3_FIELD), random.randrange(1 << 30), "r2_models2.json"), "v3"),
        "v2r1": sdk(lambda: F.R2ModelBot(None, random.randrange(1 << 30)), "v2r1"),
        "v2fin": sdk(lambda: F.R2ModelBot(random.choice(F.R2_FINALISTS), random.randrange(1 << 30)), "v2fin"),
        "r_mw": sdk(lambda: F.R1ModelBot("mw", random.randrange(1 << 30)), "r_mw"),
        "r_shr": sdk(lambda: F.R1ModelBot("Shresth Rana", random.randrange(1 << 30)), "r_shr"),
        "r_dak": sdk(lambda: F.R1ModelBot("Daksh Malhan", random.randrange(1 << 30)), "r_dak"),
        "r_thr": sdk(lambda: F.R1ModelBot("thearly man", random.randrange(1 << 30)), "r_thr"),
        "r_hen": sdk(lambda: F.R1ModelBot("Henrik Gombos", random.randrange(1 << 30)), "r_hen"),
        "r_lor": sdk(lambda: F.R1ModelBot("Lorenzo P.", random.randrange(1 << 30)), "r_lor"),
        "r_kas": sdk(lambda: F.R1ModelBot("Kashyap Patel", random.randrange(1 << 30)), "r_kas"),
        "r_suj": sdk(lambda: F.R1ModelBot("Sujay", random.randrange(1 << 30)), "r_suj"),
        "r_kri": sdk(lambda: F.R1ModelBot("krishnah37", random.randrange(1 << 30)), "r_kri"),
        "r_sco": sdk(lambda: F.R1ModelBot("Scott Middlecamp", random.randrange(1 << 30)), "r_sco"),
        "r_vai": sdk(lambda: F.R1ModelBot("Vaibhaav Tiwari", random.randrange(1 << 30)), "r_vai"),
        "r_sam": sdk(lambda: F.R1ModelBot("Sam Thomas", random.randrange(1 << 30)), "r_sam"),
        "r_moh": sdk(lambda: F.R1ModelBot("mohansriram", random.randrange(1 << 30)), "r_moh"),
        "r_ved": sdk(lambda: F.R1ModelBot("Vedant Bhardwaj", random.randrange(1 << 30)), "r_ved"),
        "r_vid": sdk(lambda: F.R1ModelBot("vidvuds calitis", random.randrange(1 << 30)), "r_vid"),
        "r_eli": sdk(lambda: F.R1ModelBot("Elias Marlowe", random.randrange(1 << 30)), "r_eli"),
        "r_osk": sdk(lambda: F.R1ModelBot("Oskar Behzadpour", random.randrange(1 << 30)), "r_osk"),
        "example": sdk(lambda: F.TrainedExampleBot(seed=random.randrange(1 << 30)), "example"),
        "tight": sdk(F.StarterTightBot, "tight"),
        "aggro": sdk(lambda: F.StarterAggroBot(seed=random.randrange(1 << 30)), "aggro"),
        "potodds": sdk(lambda: F.EquityPotOddsBot(seed=random.randrange(1 << 30)), "potodds"),
        # Gemini's archetypes / rivals
        "icm_exploit": lambda: LLM_ICM_ExploiterBot(),
        "cfr_gto": lambda: LLM_CFR_GTO_Bot(),
        "trap": lambda: LLM_TrapCounterBot(),
        "shark": lambda: SharkTAGBot(),
        "heur": lambda: HeuristicEquityBot(),
        "nit": lambda: TightPassiveNitBot(),
        "station": lambda: CallingStationBot(),
        "afk": lambda: TimeoutFoldBot(),
        "maniac": lambda: ManiacAggroBot(seed=random.randrange(1 << 30)),
        "neural_lag": lambda: RivalNeuralLAGBot(),
        "bayes": lambda: RivalBayesianProfilerBot(),
        "claude": lambda: RivalClaudeStyleBot(),
        "sniper": lambda: RivalICMSniperBot(),
    }
    def _submission():
        import submission_bot
        return SDKBotDriver(submission_bot.MyBot(), "submission")

    def _fallback():
        import importlib.util
        spec = importlib.util.spec_from_file_location("fallback_bot", "contest/submission_fallback_apexbot.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return SDKBotDriver(mod.MyBot(), "fallback")

    reg["submission"] = _submission
    reg["fallback"] = _fallback
    try:
        from bots import challenger  # optional: the bot being developed
        reg.update(challenger.registry())
    except ImportError:
        pass
    return reg


FIELDS: Dict[str, List[str]] = {
    "starter": ["starter"] * 7,
    "v3": ["v3"] * 7,
    "v4": ["r4_mw", "r4_dak", "r4_vai", "r4_lor", "r4_hen", "r4_shr", "r4_kas"],
    "v5": ["r5_mw", "r5_dak", "r5_vai", "r5_lor", "r5_hen", "r5_shr", "r5_kas"],
    "v3advjam": ["advjam", "advjam", "v3", "v3", "v3", "v3", "v3"],
    "v3adv4": ["adv4b", "adv4b", "v3", "v3", "v3", "v3", "v3"],
    "v3advraise": ["advraise", "advraise", "v3", "v3", "v3", "v3", "v3"],
    "v3adv3": ["adv3b", "adv3b", "v3", "v3", "v3", "v3", "v3"],
    "v3advcall": ["advcall", "advcall", "v3", "v3", "v3", "v3", "v3"],
    "v3mirror": ["v3"] * 6 + ["@hero"],
    "v2final": ["v2fin"] * 7,
    "v2r1": ["v2r1"] * 7,
    "v2mirror": ["v2fin"] * 6 + ["@hero"],
    "r1top": ["r_mw", "r_shr", "r_dak", "r_thr", "r_hen", "r_lor", "r_kas"],
    "r1mix": ["r_mw", "r_dak", "r_shr", "r_suj", "r_kri", "r_sco", "r_eli"],
    "r1mid": ["r_suj", "r_kri", "r_sco", "r_vai", "r_sam", "r_ved", "r_vid"],
    "r1mw": ["r_mw", "r_mw", "r_dak", "r_dak", "r_shr", "r_thr", "r_hen"],
    "r1self": ["r_osk", "r_mw", "r_shr", "r_dak", "r_thr", "r_hen", "r_lor"],
    "example": ["example"] * 7,
    "sdkmix": ["starter", "starter", "example", "tight", "aggro", "potodds", "starter2"],
    "starter2": ["starter2"] * 7,
    "replay": ["m_devin", "m_henrik", "m_scott", "m_lorenzo", "m_evan", "m_vidvuds", "m_shresth"],
    "replay2": ["m_scott", "m_evan", "m_shresth", "m_henrik", "@hero", "starter", "m_devin"],
    "entrants": ["lr30", "lr50", "lr70", "lr50", "starter", "neural_lag", "@hero"],
    "entrants2": ["lr30", "lr50", "lr70", "starter2", "starter", "icm_exploit", "lr50"],
    "starters": ["starter", "starter2", "starter", "starter2", "starter", "starter2", "example"],
    "realistic": ["starter", "example", "tight", "aggro", "icm_exploit", "shark", "neural_lag"],
    "llm": ["icm_exploit", "cfr_gto", "trap", "shark", "neural_lag", "bayes", "claude"],
    "round1": ["icm_exploit", "cfr_gto", "shark", "heur", "nit", "station", "afk"],
    "mirror": ["@hero", "@hero", "@hero", "starter", "example", "icm_exploit", "neural_lag"],
}


def _hero_factory(reg: Dict[str, Callable[[], object]], spec: str) -> Callable[[], object]:
    """'apex_sdk' or 'apex_sdk:P_OPEN_BASE=0.4,P_PUSH_ZONE_BB=15' (ApexBot param overrides)."""
    if spec.startswith("file:"):
        import importlib.util
        from pokerbot_sdk import SDKBotDriver
        path = spec[5:]
        m = importlib.util.module_from_spec(importlib.util.spec_from_file_location("hero_%d" % abs(hash(path)), path))
        m.__spec__.loader.exec_module(m)
        return lambda: SDKBotDriver(m.MyBot(), "file_hero")
    if ":" not in spec:
        return reg[spec]
    base, params = spec.split(":", 1)
    over = {k: float(v) for k, v in (kv.split("=") for kv in params.split(","))}
    from bots.apex_bot import ApexBot
    from bots.apex_plus import ApexPlus
    from bots.sdk_adapter import MyBot
    from pokerbot_sdk import SDKBotDriver
    core_cls = type("Tuned", (ApexPlus if base.startswith("plus") else ApexBot,), over)
    if base.endswith("_raw"):
        return lambda: core_cls()
    return lambda: SDKBotDriver(MyBot(core_cls()), "apex_tuned")


def _run_chunk(args: Tuple[str, str, List[int]]) -> Tuple[List[Tuple[int, float]], Dict[str, Tuple[int, int, int, int]]]:
    hero, field, seeds = args
    from poker_engine.simulator import SitAndGoTable
    reg = _registry()
    reg[hero] = _hero_factory(reg, hero)
    names = [hero] + [hero if n == "@hero" else n for n in FIELDS[field]]
    pts: List[Tuple[int, float]] = []
    lat: Dict[str, Tuple[int, int, int, int]] = {}
    from poker_engine.simulator import BotLatencyStats
    stats: Dict[str, BotLatencyStats] = {}
    for sd in seeds:
        random.seed(sd * 7919 + 1)
        rng = random.Random(sd)
        bots = [reg[n]() for n in names]
        hero_bot = bots[0]
        try:
            hero_bot.name = "HERO"
        except Exception:
            pass
        order = list(range(8))
        rng.shuffle(order)
        seated = [bots[i] for i in order]
        res = SitAndGoTable(seated, seed=sd, latency_stats=stats).play_tournament()
        pts.append((sd, res.points[order.index(0)]))
    h = stats.get("HERO")
    if h:
        lat["HERO"] = (h.decisions, h.total_ns, h.max_ns, h.timeouts + h.errors)
    return pts, lat


def run_points(hero: str, field: str, tables: int, seed: int = 1, workers: int = 0,
               pool: "ProcessPoolExecutor | None" = None) -> List[float]:
    """Hero points per table, ordered by table seed (for paired comparisons)."""
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    seeds = list(range(seed * 100000, seed * 100000 + tables))
    chunks = [(hero, field, seeds[i::workers]) for i in range(workers) if seeds[i::workers]]
    out: List[Tuple[int, float]] = []
    ex = pool or ProcessPoolExecutor(workers)
    try:
        for pts, _lat in ex.map(_run_chunk, chunks):
            out.extend(pts)
    finally:
        if pool is None:
            ex.shutdown()
    out.sort()
    return [p for _, p in out]


def bench(hero: str, field: str, tables: int, seed: int = 1, workers: int = 0) -> Tuple[float, float]:
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    seeds = list(range(seed * 100000, seed * 100000 + tables))
    chunks = [(hero, field, seeds[i::workers]) for i in range(workers) if seeds[i::workers]]
    all_pts: List[float] = []
    dec = tot = mx = bad = 0
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for pts, lat in ex.map(_run_chunk, chunks):
            all_pts.extend(p for _, p in pts)
            if "HERO" in lat:
                d, t, m, b = lat["HERO"]
                dec += d; tot += t; mx = max(mx, m); bad += b
    n = len(all_pts)
    mean = sum(all_pts) / n
    var = sum((p - mean) ** 2 for p in all_pts) / max(1, n - 1)
    se = math.sqrt(var / n)
    win = sum(1 for p in all_pts if p >= 9.99) / n
    itm = sum(1 for p in all_pts if p > 0.5) / n
    print(f"{hero:>14} vs {field:<10} {mean:6.3f} ± {se:5.3f}  win {win:5.1%}  itm {itm:5.1%}  "
          f"lat {tot / max(1, dec) / 1000:6.1f}us max {mx / 1000:7.0f}us bad {bad}  ({time.time() - t0:.0f}s)",
          flush=True)
    return mean, se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", default="apex_sdk")
    ap.add_argument("--field", default="sdkmix")
    ap.add_argument("--tables", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    heroes = a.hero.split("|") if "|" in a.hero else a.hero.split(",")
    fields = list(FIELDS) if a.field == "all" else a.field.split(",")
    for f in fields:
        for h in heroes:
            bench(h, f, a.tables, a.seed, a.workers)


if __name__ == "__main__":
    main()
