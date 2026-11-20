"""
Tournament Simulation & Benchmarking Suite for Deep-ML Pokerbot Tournament.
Runs multi-table 8-handed Sit-and-Go tournaments and full Elimination Bracket simulations,
reporting:
- Average Points per Table (out of 10/5/2 payout, random baseline = 17/8 = 2.125 pts/table)
- Top-3 ITM (In-The-Money) Finish Rate (baseline = 37.5%)
- 1st Place Win Rate (baseline = 12.5%)
- Per-decision Latency (Mean, P99, Max in microseconds, and Timeout/Error counts)
"""

from __future__ import annotations
import argparse
import time
from typing import Callable, Dict, List, Sequence, Tuple

from bots.apex_bot import ApexBot
from bots.archetypes import (
    CallingStationBot,
    HeuristicEquityBot,
    LLM_CFR_GTO_Bot,
    LLM_ICM_ExploiterBot,
    LLM_TrapCounterBot,
    ManiacAggroBot,
    SharkTAGBot,
    TightPassiveNitBot,
    TimeoutFoldBot,
)
from bots.rival_bayesian_profiler import RivalBayesianProfilerBot
from bots.rival_cfr_pro import RivalCFRProBot
from bots.rival_claude_style import RivalClaudeStyleBot
from bots.rival_gpt_oss_style import RivalGPTOSSStyleBot
from bots.rival_icm_sniper import RivalICMSniperBot
from bots.rival_monte_carlo_ev import RivalMonteCarloEVBot
from bots.rival_neural_lag import RivalNeuralLAGBot
from poker_engine.simulator import (
    AGGRESSIVE_BLIND_SCHEDULE,
    DEFAULT_BLIND_SCHEDULE,
    BotLatencyStats,
    SitAndGoTable,
)


def run_benchmark(
    num_tables: int = 200,
    schedule_name: str = "default",
    field_mode: str = "mixed",
    seed: int = 100,
) -> None:
    schedule = (
        AGGRESSIVE_BLIND_SCHEDULE if schedule_name == "aggressive" else DEFAULT_BLIND_SCHEDULE
    )

    if field_mode == "round1":
        # Realistic Round 1 field including prepared LLM bots + classical archetypes
        factories: List[Tuple[str, Callable[[], object]]] = [
            ("ApexBot_v3", lambda: ApexBot()),
            ("LLM_ICM_Exploiter", lambda: LLM_ICM_ExploiterBot()),
            ("LLM_CFR_GTO", lambda: LLM_CFR_GTO_Bot()),
            ("SharkTAGBot", lambda: SharkTAGBot()),
            ("HeuristicEquity", lambda: HeuristicEquityBot()),
            ("TightPassiveNit", lambda: TightPassiveNitBot()),
            ("CallingStation", lambda: CallingStationBot()),
            ("TimeoutFold", lambda: TimeoutFoldBot()),
        ]
    elif field_mode == "opus_meta":
        # Heavy Prepared-LLM / Finals Meta: Every opponent is a well-prepared LLM-authored
        # or strong tournament strategy bot!
        factories = [
            ("ApexBot_v3", lambda: ApexBot()),
            ("LLM_ICM_Exploit_1", lambda: LLM_ICM_ExploiterBot()),
            ("LLM_ICM_Exploit_2", lambda: LLM_ICM_ExploiterBot()),
            ("LLM_CFR_GTO_1", lambda: LLM_CFR_GTO_Bot()),
            ("LLM_CFR_GTO_2", lambda: LLM_CFR_GTO_Bot()),
            ("LLM_TrapCounter", lambda: LLM_TrapCounterBot()),
            ("SharkTAGBot", lambda: SharkTAGBot()),
            ("HeuristicEquity", lambda: HeuristicEquityBot()),
        ]
    elif field_mode == "rivals":
        # Direct 8-Player Battle Royale: ApexBot vs all 7 Independent Subagent Rival Bots!
        factories = [
            ("ApexBot", lambda: ApexBot()),
            ("Rival_NeuralLAG", lambda: RivalNeuralLAGBot()),
            ("Rival_MonteCarloEV", lambda: RivalMonteCarloEVBot()),
            ("Rival_ICMSniper", lambda: RivalICMSniperBot()),
            ("Rival_BayesianProf", lambda: RivalBayesianProfilerBot()),
            ("Rival_CFRPro", lambda: RivalCFRProBot()),
            ("Rival_ClaudeStyle", lambda: RivalClaudeStyleBot()),
            ("Rival_GPTOSSStyle", lambda: RivalGPTOSSStyleBot()),
        ]
    elif field_mode == "finals":
        factories = [
            ("ApexBot_1", lambda: ApexBot()),
            ("ApexBot_2", lambda: ApexBot()),
            ("LLM_ICM_Exploiter", lambda: LLM_ICM_ExploiterBot()),
            ("LLM_CFR_GTO", lambda: LLM_CFR_GTO_Bot()),
            ("LLM_TrapCounter", lambda: LLM_TrapCounterBot()),
            ("SharkTAG_1", lambda: SharkTAGBot()),
            ("SharkTAG_2", lambda: SharkTAGBot()),
            ("HeuristicEquity", lambda: HeuristicEquityBot()),
        ]
    else:
        factories = [
            ("ApexBot_v3", lambda: ApexBot()),
            ("LLM_ICM_Exploiter", lambda: LLM_ICM_ExploiterBot()),
            ("LLM_CFR_GTO", lambda: LLM_CFR_GTO_Bot()),
            ("LLM_TrapCounter", lambda: LLM_TrapCounterBot()),
            ("SharkTAGBot", lambda: SharkTAGBot()),
            ("HeuristicEquity", lambda: HeuristicEquityBot()),
            ("TightPassiveNit", lambda: TightPassiveNitBot()),
            ("ManiacAggro", lambda: ManiacAggroBot(seed=seed)),
        ]

    names = [name for name, _ in factories]
    total_points: Dict[str, float] = {n: 0.0 for n in names}
    first_places: Dict[str, int] = {n: 0 for n in names}
    top3_finishes: Dict[str, int] = {n: 0 for n in names}
    latency_stats: Dict[str, BotLatencyStats] = {}
    cap_hits = 0
    total_hands = 0

    import random
    seat_rng = random.Random(seed + 99999)
    t0 = time.perf_counter()
    for t_idx in range(num_tables):
        # Randomly permute seat assignments every table to eliminate neighbor/position bias
        rotated = list(factories)
        seat_rng.shuffle(rotated)
        bots = []
        seat_names = []
        for b_name, fn in rotated:
            bot = fn()
            bot.name = b_name
            bots.append(bot)
            seat_names.append(b_name)

        table = SitAndGoTable(
            bots=bots,
            blind_schedule=schedule,
            enforce_1ms=True,
            seed=seed + t_idx * 97,
            latency_stats=latency_stats,
        )
        res = table.play_tournament()
        if res.hit_hand_cap:
            cap_hits += 1
        total_hands += res.hands_played

        # Rank seats by points earned on this table
        ranked_seats = sorted(range(8), key=lambda i: res.points[i], reverse=True)
        for i in range(8):
            b_name = seat_names[i]
            pts = res.points[i]
            total_points[b_name] += pts
            if pts >= 1.5:  # Top-3 score
                top3_finishes[b_name] += 1
        first_places[seat_names[ranked_seats[0]]] += 1

    elapsed = time.perf_counter() - t0
    print("=" * 88)
    print(
        f"DEEP-ML SIT-AND-GO BENCHMARK | Tables: {num_tables} | Field: {field_mode} | "
        f"Blinds: {schedule_name}"
    )
    print(
        f"Simulated {total_hands:,} hands in {elapsed:.2f}s ({total_hands/elapsed:,.0f} hands/sec) | "
        f"200-Hand Cap Reached: {cap_hits}/{num_tables} ({100*cap_hits/num_tables:.1f}%)"
    )
    print("=" * 88)
    print(
        f"{'Bot Name':<20} | {'Avg Pts':>8} | {'ROI vs Avg':>10} | {'Top-3 %':>8} | "
        f"{'1st %':>7} | {'Mean us':>8} | {'P99 us':>8} | {'Timeouts':>8}"
    )
    print("-" * 88)

    sorted_names = sorted(names, key=lambda n: total_points[n], reverse=True)
    for n in sorted_names:
        avg_pts = total_points[n] / num_tables
        roi = (avg_pts / 2.125 - 1.0) * 100.0
        t3_pct = 100.0 * top3_finishes[n] / num_tables
        w1_pct = 100.0 * first_places[n] / num_tables
        lstat = latency_stats.get(n, BotLatencyStats())
        print(
            f"{n:<20} | {avg_pts:>8.3f} | {roi:>+9.1f}% | {t3_pct:>7.1f}% | "
            f"{w1_pct:>6.1f}% | {lstat.mean_us:>7.1f}u | {lstat.p99_us:>7.1f}u | "
            f"{lstat.timeouts:>8d}"
        )
    print("=" * 88)


def run_bracket_simulation(
    num_brackets: int = 30,
    seed: int = 777,
) -> None:
    """
    Simulates full 64-player Elimination Brackets (Round 1: 64 -> Round 2: 32 ->
    Semifinal: 16 -> Final Table: 8 -> Top 3 Podium) where 1 entry is ApexBot
    and the other 63 entries are a realistic distribution of competition bots:
    - 12 SharkTAGBots
    - 20 HeuristicEquityBots
    - 11 TightPassiveNitBots
    - 8 CallingStationBots
    - 6 ManiacAggroBots
    - 6 TimeoutFoldBots
    """
    import random

    rng = random.Random(seed)
    rounds_reached = []
    final_table_reached = 0
    podium_finishes = 0
    tournament_wins = 0

    t0 = time.perf_counter()
    for b_idx in range(num_brackets):
        # Create 64 entrants: entrant 0 is our ApexBot
        entrants: List[Tuple[str, str]] = [("Hero_ApexBot", "ApexBot")]
        archetype_pool = (
            ["Rival_NeuralLAG"] * 5
            + ["Rival_MonteCarloEV"] * 5
            + ["Rival_ICMSniper"] * 5
            + ["Rival_BayesianProf"] * 5
            + ["Rival_CFRPro"] * 5
            + ["Rival_ClaudeStyle"] * 4
            + ["Rival_GPTOSSStyle"] * 4
            + ["LLM_ICM_Exploiter"] * 5
            + ["LLM_CFR_GTO"] * 5
            + ["LLM_TrapCounter"] * 4
            + ["SharkTAG"] * 5
            + ["HeuristicEquity"] * 4
            + ["TightPassive"] * 3
            + ["CallingStation"] * 2
            + ["Maniac"] * 1
            + ["TimeoutFold"] * 1
        )
        for i, arch in enumerate(archetype_pool):
            entrants.append((f"{arch}_{i+1}", arch))

        def make_bot(arch_type: str, s: int):
            if arch_type == "ApexBot":
                return ApexBot()
            if arch_type == "Rival_NeuralLAG":
                return RivalNeuralLAGBot()
            if arch_type == "Rival_MonteCarloEV":
                return RivalMonteCarloEVBot()
            if arch_type == "Rival_ICMSniper":
                return RivalICMSniperBot()
            if arch_type == "Rival_BayesianProf":
                return RivalBayesianProfilerBot()
            if arch_type == "Rival_CFRPro":
                return RivalCFRProBot()
            if arch_type == "Rival_ClaudeStyle":
                return RivalClaudeStyleBot()
            if arch_type == "Rival_GPTOSSStyle":
                return RivalGPTOSSStyleBot()
            if arch_type == "LLM_ICM_Exploiter":
                return LLM_ICM_ExploiterBot()
            if arch_type == "LLM_CFR_GTO":
                return LLM_CFR_GTO_Bot()
            if arch_type == "LLM_TrapCounter":
                return LLM_TrapCounterBot()
            if arch_type == "SharkTAG":
                return SharkTAGBot()
            if arch_type == "HeuristicEquity":
                return HeuristicEquityBot()
            if arch_type == "TightPassive":
                return TightPassiveNitBot()
            if arch_type == "CallingStation":
                return CallingStationBot()
            if arch_type == "Maniac":
                return ManiacAggroBot(seed=s)
            return TimeoutFoldBot()

        hero_round = 1
        # 4 rounds: 64 -> 32 -> 16 -> 8 (Final Table)
        for r_num in range(1, 5):
            n_players = len(entrants)
            n_tables = n_players // 8
            scores = {name: 0.0 for name, _ in entrants}

            # Each player plays 4 Sit-and-Go tables per round to determine average points
            tables_per_round = 4
            for t_rep in range(tables_per_round):
                shuffled = list(entrants)
                rng.shuffle(shuffled)
                for tab_i in range(n_tables):
                    group = shuffled[tab_i * 8 : (tab_i + 1) * 8]
                    bots = [make_bot(arch, seed + b_idx * 100 + r_num * 10 + t_rep) for _, arch in group]
                    res = SitAndGoTable(
                        bots=bots,
                        enforce_1ms=True,
                        seed=seed + b_idx * 1000 + r_num * 100 + t_rep * 10 + tab_i,
                    ).play_tournament()
                    for seat_i, (p_name, _) in enumerate(group):
                        scores[p_name] += res.points[seat_i] / tables_per_round

            ranked = sorted(entrants, key=lambda item: scores[item[0]], reverse=True)
            if r_num < 4:
                # Top half advances
                survivors = ranked[: n_players // 2]
                if any(name == "Hero_ApexBot" for name, _ in survivors):
                    hero_round = r_num + 1
                    entrants = survivors
                else:
                    break
            else:
                # Final Table of 8!
                final_table_reached += 1
                hero_rank = next(i for i, (name, _) in enumerate(ranked) if name == "Hero_ApexBot")
                if hero_rank < 3:
                    podium_finishes += 1
                if hero_rank == 0:
                    tournament_wins += 1

        rounds_reached.append(hero_round)

    dt = time.perf_counter() - t0
    print("=" * 88)
    print(f"64-PLAYER ELIMINATION BRACKET SIMULATION ({num_brackets} Full Tournaments in {dt:.2f}s)")
    print("Baseline random expectation: Final Table = 12.5%, Top-3 Podium = 4.7%, 1st Place = 1.6%")
    print("-" * 88)
    print(f"Hero ApexBot Final Table Rate (Top 8 / 64): {final_table_reached}/{num_brackets} ({100*final_table_reached/num_brackets:.1f}%)")
    print(f"Hero ApexBot Podium Rate     (Top 3 / 64): {podium_finishes}/{num_brackets} ({100*podium_finishes/num_brackets:.1f}%)")
    print(f"Hero ApexBot Champion Rate   (1st   / 64): {tournament_wins}/{num_brackets} ({100*tournament_wins/num_brackets:.1f}%)")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=int, default=200)
    parser.add_argument("--schedule", type=str, default="default", choices=["default", "aggressive"])
    parser.add_argument(
        "--field",
        type=str,
        default="mixed",
        choices=["mixed", "round1", "finals", "opus_meta", "rivals"],
    )
    parser.add_argument("--bracket", type=int, default=0, help="Number of 64-player brackets to simulate")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.bracket > 0:
        run_bracket_simulation(num_brackets=args.bracket, seed=args.seed)
    else:
        run_benchmark(
            num_tables=args.tables,
            schedule_name=args.schedule,
            field_mode=args.field,
            seed=args.seed,
        )
