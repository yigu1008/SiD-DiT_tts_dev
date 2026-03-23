from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import sampling_unified as su
from cem_opt import CEMResult, optimize_cem
from prompt_basis import PromptBasisEmbeddings
from rollout_runner import RolloutResult, run_dynamic_rollout
from weight_policy import WeightParams


@dataclass
class PromptGAEval:
    score: float
    genome: list[int]
    subset_indices: list[int]
    subset_labels: list[str]
    blend_family: str
    theta: list[float]
    rollout: RolloutResult
    cem: CEMResult


@dataclass
class PromptGASearchResult:
    best: PromptGAEval
    history: list[dict[str, Any]]
    evaluation_log: list[dict[str, Any]]
    eval_calls_total: int
    nfe_total: int
    wallclock_total_sec: float


def _stable_seed(base_seed: int, genome: list[int]) -> int:
    h = 1469598103934665603
    for x in genome:
        h ^= int(x) + 0x9E3779B97F4A7C15
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int((int(base_seed) + h) % (2**31 - 1))


def _repair_subset(indices: list[int], pool_size: int, k: int, rng: np.random.Generator) -> list[int]:
    k = max(1, min(int(k), int(pool_size)))
    out: list[int] = []
    seen: set[int] = set()
    for x in indices:
        idx = int(x) % int(pool_size)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= k:
            break
    if len(out) < k:
        missing = [i for i in range(pool_size) if i not in seen]
        rng.shuffle(missing)
        out.extend(missing[: k - len(out)])
    return out[:k]


def decode_prompt_genome(
    genome: list[int],
    pool_size: int,
    basis_k: int,
    blend_families: list[str],
    rng: np.random.Generator,
) -> tuple[list[int], str, list[int]]:
    if len(genome) < 2:
        raise ValueError("Genome too short.")
    fams = [str(x) for x in blend_families]
    if len(fams) == 0:
        fams = ["nlerp"]
    k = max(1, min(int(basis_k), int(pool_size)))
    subset_raw = [int(x) for x in genome[:k]]
    repaired_subset = _repair_subset(subset_raw, pool_size, k, rng)
    fam_gene = int(genome[k]) if len(genome) > k else 0
    family = fams[abs(fam_gene) % len(fams)]
    repaired = list(repaired_subset) + [int(abs(fam_gene) % len(fams))]
    return repaired_subset, family, repaired


def random_prompt_genome(
    pool_size: int,
    basis_k: int,
    blend_families: list[str],
    rng: np.random.Generator,
) -> list[int]:
    fam_count = max(1, len(blend_families))
    genes = [int(rng.integers(0, pool_size)) for _ in range(max(1, basis_k))]
    genes.append(int(rng.integers(0, fam_count)))
    return genes


def rank_select(scored: list[tuple[float, list[int]]], rng: np.random.Generator, rank_pressure: float) -> list[int]:
    n = len(scored)
    if n <= 0:
        raise RuntimeError("rank_select requires non-empty scored list.")
    if n == 1:
        return list(scored[0][1])
    s = float(np.clip(float(rank_pressure), 1.0, 2.0))
    probs = np.empty((n,), dtype=np.float64)
    for idx_desc in range(n):
        rank_worst_first = n - 1 - idx_desc
        probs[idx_desc] = ((2.0 - s) / n) + (2.0 * rank_worst_first * (s - 1.0) / (n * (n - 1)))
    probs = probs / probs.sum()
    idx = int(rng.choice(np.arange(n, dtype=np.int64), p=probs))
    return list(scored[idx][1])


def tournament_select(scored: list[tuple[float, list[int]]], rng: np.random.Generator, k: int) -> list[int]:
    n = len(scored)
    if n <= 0:
        raise RuntimeError("tournament_select requires non-empty scored list.")
    kk = max(1, min(int(k), n))
    picks = [scored[int(rng.integers(0, n))] for _ in range(kk)]
    return list(max(picks, key=lambda row: row[0])[1])


def crossover_uniform(a: list[int], b: list[int], rng: np.random.Generator) -> list[int]:
    return [int(ga if rng.random() < 0.5 else gb) for ga, gb in zip(a, b)]


def mutate_prompt_genome(genome: list[int], pool_size: int, fam_count: int, p: float, rng: np.random.Generator) -> list[int]:
    out = list(genome)
    for i in range(len(out)):
        if rng.random() >= float(p):
            continue
        if i == len(out) - 1:
            out[i] = int(rng.integers(0, max(1, fam_count)))
        else:
            out[i] = int(rng.integers(0, max(1, pool_size)))
    return out


def _weight_evolution(rollout: RolloutResult) -> list[dict[str, Any]]:
    return [
        {
            "step": int(st.step),
            "sigma": float(st.sigma),
            "progress": float(st.progress),
            "blend_family": str(st.blend_family),
            "selected_indices": [int(x) for x in st.selected_indices],
            "selected_labels": [str(x) for x in st.selected_labels],
            "selected_weights": [float(x) for x in st.selected_weights],
            "weights_by_label": {str(k): float(v) for k, v in st.weights_by_label.items()},
            "preview_reward": float(st.preview_reward),
            "delta_reward": float(st.delta_reward),
        }
        for st in rollout.step_traces
    ]


def run_ga_prompt_search(
    args: Any,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    basis_emb: PromptBasisEmbeddings,
    save_best_image_path: str | None,
) -> PromptGASearchResult:
    rng = np.random.default_rng(int(seed) + 9001)
    pool_size = len(basis_emb.pe_list)
    basis_k = max(1, min(int(args.basis_k), pool_size))
    blend_families = [str(x).strip().lower() for x in args.blend_families if str(x).strip()]
    if len(blend_families) == 0:
        blend_families = ["nlerp"]
    fam_count = len(blend_families)

    pop_size = max(4, int(args.ga_population))
    elites = max(1, min(int(args.ga_elites), pop_size))
    steps = int(args.steps)

    # Anchor: first K candidates + configured family.
    anchor_family = str(args.ga_anchor_family).strip().lower()
    if anchor_family not in blend_families:
        anchor_family = blend_families[0]
    anchor_fam_gene = int(blend_families.index(anchor_family))
    anchor = list(range(basis_k)) + [anchor_fam_gene]

    population: list[list[int]] = [anchor]
    while len(population) < pop_size:
        population.append(random_prompt_genome(pool_size, basis_k, blend_families, rng))

    eval_cache: dict[tuple[int, ...], PromptGAEval] = {}
    eval_calls_total = 0
    nfe_total = 0
    history: list[dict[str, Any]] = []
    evaluation_log: list[dict[str, Any]] = []

    best_eval: PromptGAEval | None = None
    search_start = time.perf_counter()

    def evaluate(genome: list[int], tag_prefix: str) -> PromptGAEval:
        nonlocal eval_calls_total, nfe_total
        local_rng = np.random.default_rng(_stable_seed(seed, genome))
        subset_indices, family, repaired = decode_prompt_genome(
            genome,
            pool_size=pool_size,
            basis_k=basis_k,
            blend_families=blend_families,
            rng=local_rng,
        )
        key = tuple(int(x) for x in repaired)
        if bool(args.ga_eval_cache) and key in eval_cache:
            cached = eval_cache[key]
            # cache hit: still count no new eval calls.
            return cached

        k = len(subset_indices)

        def objective(theta_vec: np.ndarray) -> tuple[float, RolloutResult]:
            theta = WeightParams.from_vector(theta_vec, k)
            trace = run_dynamic_rollout(
                args=args,
                ctx=ctx,
                reward_ctx=reward_ctx,
                prompt=prompt,
                metadata=metadata,
                seed=seed,
                h=h,
                w=w,
                orig_h=orig_h,
                orig_w=orig_w,
                basis_emb=basis_emb,
                basis_indices=subset_indices,
                blend_family=family,
                weight_params=theta,
                preview_every=int(args.preview_every),
                save_path=None,
                tag=f"{tag_prefix}_{family}",
            )
            return float(trace.final_score), trace

        cem = optimize_cem(
            objective=objective,
            dim=2 * k,
            seed=_stable_seed(seed + 1777, repaired),
            n_iters=int(args.cem_iters),
            pop_size=int(args.cem_population),
            elite_frac=float(args.cem_elite_frac),
            init_std=float(args.cem_init_std),
            min_std=float(args.cem_min_std),
            clip_value=float(args.cem_clip),
        )
        eval_calls_total += int(cem.eval_calls)
        nfe_total += int(cem.eval_calls * steps)

        theta = [float(x) for x in cem.best_x.tolist()]
        rollout = cem.best_aux
        result = PromptGAEval(
            score=float(cem.best_score),
            genome=[int(x) for x in repaired],
            subset_indices=[int(i) for i in subset_indices],
            subset_labels=[basis_emb.labels[i] for i in subset_indices],
            blend_family=family,
            theta=theta,
            rollout=rollout,
            cem=cem,
        )
        evaluation_log.append(
            {
                "tag": str(tag_prefix),
                "score": float(result.score),
                "genome": [int(x) for x in result.genome],
                "subset_indices": [int(x) for x in result.subset_indices],
                "subset_labels": [str(x) for x in result.subset_labels],
                "blend_family": str(result.blend_family),
                "theta": [float(x) for x in result.theta],
                "preview_rewards": [float(x) for x in result.rollout.preview_rewards],
                "step_weight_evolution": _weight_evolution(result.rollout),
                "cem_history": list(result.cem.history),
            }
        )
        if bool(args.ga_eval_cache):
            eval_cache[key] = result
        return result

    for gen in range(int(args.ga_generations)):
        gen_start = time.perf_counter()
        evaluated: list[PromptGAEval] = []
        for i, genome in enumerate(population):
            evaluated.append(evaluate(genome, tag_prefix=f"ga_g{gen}_i{i}"))
        evaluated.sort(key=lambda row: row.score, reverse=True)

        if best_eval is None or evaluated[0].score > best_eval.score:
            best_eval = evaluated[0]
            if save_best_image_path:
                best_theta = WeightParams.from_vector(np.asarray(best_eval.theta, dtype=np.float64), len(best_eval.subset_indices))
                saved_trace = run_dynamic_rollout(
                    args=args,
                    ctx=ctx,
                    reward_ctx=reward_ctx,
                    prompt=prompt,
                    metadata=metadata,
                    seed=seed,
                    h=h,
                    w=w,
                    orig_h=orig_h,
                    orig_w=orig_w,
                    basis_emb=basis_emb,
                    basis_indices=best_eval.subset_indices,
                    blend_family=best_eval.blend_family,
                    weight_params=best_theta,
                    preview_every=int(args.preview_every),
                    save_path=save_best_image_path,
                    tag="ga_best_save",
                )
                best_eval.rollout = saved_trace

        topk = max(1, int(args.ga_log_topk))
        history.append(
            {
                "generation": int(gen),
                "best_score": float(evaluated[0].score),
                "mean_score": float(np.mean([row.score for row in evaluated])),
                "eval_calls_total": int(eval_calls_total),
                "nfe_per_generation": int(sum(row.cem.eval_calls for row in evaluated) * steps),
                "nfe_total": int(nfe_total),
                "wallclock_sec": float(time.perf_counter() - gen_start),
                "top": [
                    {
                        "rank": int(r + 1),
                        "score": float(row.score),
                        "genome": [int(x) for x in row.genome],
                        "subset_indices": [int(x) for x in row.subset_indices],
                        "subset_labels": list(row.subset_labels),
                        "blend_family": str(row.blend_family),
                        "theta": [float(x) for x in row.theta],
                        "preview_rewards": [float(x) for x in row.rollout.preview_rewards],
                        "step_weight_evolution": _weight_evolution(row.rollout),
                        "cem_history": list(row.cem.history),
                    }
                    for r, row in enumerate(evaluated[:topk])
                ],
            }
        )

        if gen + 1 >= int(args.ga_generations):
            break

        scored_pairs = [(row.score, row.genome) for row in evaluated]
        next_population: list[list[int]] = [list(row.genome) for row in evaluated[:elites]]
        while len(next_population) < pop_size:
            if str(args.ga_selection).lower() == "rank":
                pa = rank_select(scored_pairs, rng, float(args.ga_rank_pressure))
                pb = rank_select(scored_pairs, rng, float(args.ga_rank_pressure))
            else:
                pa = tournament_select(scored_pairs, rng, int(args.ga_tournament_k))
                pb = tournament_select(scored_pairs, rng, int(args.ga_tournament_k))
            child = crossover_uniform(pa, pb, rng)
            child = mutate_prompt_genome(child, pool_size, fam_count, float(args.ga_mutation_prob), rng)
            next_population.append(child)
        population = next_population

    if best_eval is None:
        raise RuntimeError("GA search produced no evaluations.")
    return PromptGASearchResult(
        best=best_eval,
        history=history,
        evaluation_log=evaluation_log,
        eval_calls_total=int(eval_calls_total),
        nfe_total=int(nfe_total),
        wallclock_total_sec=float(time.perf_counter() - search_start),
    )
