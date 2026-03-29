"""Benchmark suite for severe centered and shifted hyperbolic fixed-point tests.

Author: David Ariza Ruiz
Copyright (c) 2026 David Ariza Ruiz
Companion code for reproducible numerical experiments in the Poincare disk.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Poincare disk geometry and benchmark operators
# ============================================================

def clip_to_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n >= 1.0:
        return v * ((1.0 - eps) / max(n, 1e-15))
    return v


def project_to_ball(v: np.ndarray, rho: float, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= rho:
        return v
    return v * ((rho - eps) / max(n, 1e-15))


def mobius_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = clip_to_unit(u)
    v = clip_to_unit(v)
    uv = float(np.dot(u, v))
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    den = 1.0 + 2.0 * uv + uu * vv
    num = ((1.0 + 2.0 * uv + vv) * u) + ((1.0 - uu) * v)
    return clip_to_unit(num / max(den, 1e-15))


def mobius_neg(u: np.ndarray) -> np.ndarray:
    return -u


def mobius_scalar(r: float, x: np.ndarray) -> np.ndarray:
    nx = float(np.linalg.norm(x))
    if nx < 1e-15:
        return np.zeros_like(x)
    coeff = math.tanh(r * math.atanh(min(nx, 1.0 - 1e-15)))
    return clip_to_unit(coeff * x / nx)


def geodesic_interp(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    t = float(min(max(t, 0.0), 1.0))
    return mobius_add(x, mobius_scalar(t, mobius_add(mobius_neg(x), y)))


def midpoint_mobius(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return geodesic_interp(u, v, 0.5)


def poincare_dist(u: np.ndarray, v: np.ndarray) -> float:
    u = clip_to_unit(u)
    v = clip_to_unit(v)
    diff2 = float(np.dot(u - v, u - v))
    den = (1.0 - float(np.dot(u, u))) * (1.0 - float(np.dot(v, v)))
    arg = 1.0 + 2.0 * diff2 / max(den, 1e-15)
    return math.acosh(max(1.0, arg))


def rotate_half_plane(z: np.ndarray) -> np.ndarray:
    x, y = float(z[0]), float(z[1])
    if x >= 0.0:
        return np.array([-y, x], dtype=float)
    return np.array([y, -x], dtype=float)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    gamma: float = 0.55
    zstar_x: float = 0.0
    zstar_y: float = 0.0

    @property
    def zstar(self) -> np.ndarray:
        return np.array([self.zstar_x, self.zstar_y], dtype=float)

    def validate(self) -> None:
        if not (0.0 < self.gamma < 1.0):
            raise ValueError('gamma must lie in (0,1).')
        if float(np.linalg.norm(self.zstar)) >= 1.0:
            raise ValueError('zstar must lie in the open unit disk.')


def central_operator(z: np.ndarray, gamma: float) -> np.ndarray:
    return mobius_scalar(gamma, rotate_half_plane(z))


def benchmark_operator(z: np.ndarray, bench: BenchmarkSpec) -> np.ndarray:
    bench.validate()
    z = clip_to_unit(z)
    zstar = bench.zstar
    if float(np.linalg.norm(zstar)) < 1e-15:
        return central_operator(z, bench.gamma)
    local = mobius_add(mobius_neg(zstar), z)
    moved = central_operator(local, bench.gamma)
    return mobius_add(zstar, moved)


def residual(z: np.ndarray, bench: BenchmarkSpec) -> float:
    return poincare_dist(z, benchmark_operator(z, bench))


# ============================================================
# Sampling, mutation, and initial conditions
# ============================================================

def sample_uniform_refresh(rng: np.random.Generator, rho: float) -> np.ndarray:
    theta = rng.uniform(-math.pi, math.pi)
    r = rho * math.sqrt(rng.uniform())
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


def sample_beta_refresh(rng: np.random.Generator, rho: float) -> np.ndarray:
    theta = rng.uniform(-math.pi, math.pi)
    r = rho * math.sqrt(rng.beta(1.1, 0.8))
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


def sample_center_refresh(rng: np.random.Generator, rho: float) -> np.ndarray:
    theta = rng.uniform(-math.pi, math.pi)
    r = rho * (1.0 - rng.power(2.5))
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


def sample_boundary_refresh(rng: np.random.Generator, rho: float) -> np.ndarray:
    theta = rng.uniform(-math.pi, math.pi)
    r = rho * math.sqrt(1.0 - rng.beta(3.0, 1.4))
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


REFRESH_SAMPLERS: Dict[str, Callable[[np.random.Generator, float], np.ndarray]] = {
    'uniform': sample_uniform_refresh,
    'beta': sample_beta_refresh,
    'center': sample_center_refresh,
    'boundary': sample_boundary_refresh,
}


@dataclass(frozen=True)
class InitSpec:
    rho_work: float = 0.93
    init_rmin_factor: float = 0.68
    init_angle_min: float = -math.pi / 3.0
    init_angle_max: float = math.pi / 3.0

    def validate(self) -> None:
        if not (0.0 < self.rho_work < 1.0):
            raise ValueError('rho_work must lie in (0,1).')
        if not (0.0 <= self.init_rmin_factor <= 1.0):
            raise ValueError('init_rmin_factor must lie in [0,1].')
        if self.init_angle_min >= self.init_angle_max:
            raise ValueError('Invalid angular window.')


def sample_initial_point(rng: np.random.Generator, init: InitSpec) -> np.ndarray:
    init.validate()
    theta = rng.uniform(init.init_angle_min, init.init_angle_max)
    r = rng.uniform(init.init_rmin_factor * init.rho_work, init.rho_work)
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


def init_population(rng: np.random.Generator, init: InitSpec, pop_size: int) -> List[np.ndarray]:
    return [sample_initial_point(rng, init) for _ in range(pop_size)]


def local_mutation(z: np.ndarray, rho_work: float, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=2)
    nd = float(np.linalg.norm(direction))
    direction = direction / (nd if nd > 1e-15 else 1.0)
    jump = min(float(rng.exponential(scale=0.18)), 0.9)
    child = mobius_add(z, mobius_scalar(jump, 0.15 * direction))
    return project_to_ball(child, rho_work)


# ============================================================
# Algorithms
# ============================================================

@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 60
    generations: int = 120
    mutation_prob: float = 0.35
    refresh_eps: float = 0.10
    refresh_law: str = 'uniform'
    tournament_size: int = 2
    target: float = 1e-3

    def validate(self) -> None:
        if self.pop_size < 2:
            raise ValueError('pop_size must be at least 2.')
        if self.generations < 1:
            raise ValueError('generations must be at least 1.')
        if not (0.0 <= self.mutation_prob <= 1.0):
            raise ValueError('mutation_prob must lie in [0,1].')
        if not (0.0 <= self.refresh_eps <= 1.0):
            raise ValueError('refresh_eps must lie in [0,1].')
        if self.refresh_law not in REFRESH_SAMPLERS:
            raise ValueError(f'Unknown refresh_law={self.refresh_law!r}.')
        if self.tournament_size < 2:
            raise ValueError('tournament_size must be at least 2.')
        if self.target <= 0.0:
            raise ValueError('target must be positive.')

    @property
    def eval_budget(self) -> int:
        return self.pop_size * self.generations


@dataclass(frozen=True)
class DeterministicConfig:
    target: float = 1e-3
    alpha_mode: str = 'harmonic'

    def validate(self) -> None:
        if self.target <= 0.0:
            raise ValueError('target must be positive.')
        if self.alpha_mode not in {'harmonic'}:
            raise ValueError('Unsupported alpha_mode.')

    def alpha(self, n: int) -> float:
        if self.alpha_mode == 'harmonic':
            return 1.0 / (n + 1.0)
        raise RuntimeError('Unsupported alpha schedule.')


@dataclass(frozen=True)
class Scenario:
    family: str
    benchmark: str
    method: str
    label: str
    benchmark_spec: BenchmarkSpec
    init_spec: InitSpec
    ga_config: Optional[GAConfig] = None
    det_config: Optional[DeterministicConfig] = None


@dataclass
class RunResult:
    family: str
    benchmark: str
    method: str
    label: str
    seed: int
    final_residual: float
    hit_evaluations: int
    success: int
    wall_seconds: float
    mean_residual: float
    auc_log_residual: float
    x0_norm: float
    curve_eval: np.ndarray


def tournament(pop: List[np.ndarray], fit: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    best_local = int(idx[np.argmin(fit[idx])])
    return pop[best_local]


def expand_generation_curve_to_eval_curve(gen_curve: np.ndarray, pop_size: int) -> np.ndarray:
    pieces = [np.full(pop_size, value, dtype=float) for value in gen_curve]
    return np.concatenate(pieces, axis=0)


def run_ga(bench: BenchmarkSpec, init: InitSpec, cfg: GAConfig, seed: int) -> RunResult:
    cfg.validate()
    rng = np.random.default_rng(seed)
    refresh_sampler = REFRESH_SAMPLERS[cfg.refresh_law]

    pop = init_population(rng, init, cfg.pop_size)
    history: List[float] = []
    t0 = time.perf_counter()
    hit_evaluations = cfg.eval_budget + 1

    for g in range(cfg.generations):
        fit = np.array([residual(z, bench) for z in pop], dtype=float)
        bi = int(np.argmin(fit))
        best = pop[bi].copy()
        bestf = float(fit[bi])
        history.append(bestf)
        if hit_evaluations == cfg.eval_budget + 1 and bestf <= cfg.target:
            hit_evaluations = g * cfg.pop_size + 1

        new_pop: List[np.ndarray] = [best]
        while len(new_pop) < cfg.pop_size:
            p1 = tournament(pop, fit, cfg.tournament_size, rng)
            p2 = tournament(pop, fit, cfg.tournament_size, rng)
            child = midpoint_mobius(p1, p2)
            if rng.random() < cfg.mutation_prob:
                if rng.random() < cfg.refresh_eps:
                    child = refresh_sampler(rng, init.rho_work)
                else:
                    child = local_mutation(child, init.rho_work, rng)
            new_pop.append(project_to_ball(child, init.rho_work))
        pop = new_pop

    elapsed = time.perf_counter() - t0
    gen_curve = np.array(history, dtype=float)
    eval_curve = expand_generation_curve_to_eval_curve(gen_curve, cfg.pop_size)
    auc_log = float(np.trapezoid(np.log10(np.maximum(eval_curve, 1e-14)), dx=1.0))
    return RunResult(
        family='benchmark_suite',
        benchmark=bench.name,
        method=f'GA-{cfg.refresh_eps:.2f}',
        label=f'{bench.name}|GA-{cfg.refresh_eps:.2f}',
        seed=seed,
        final_residual=float(eval_curve[-1]),
        hit_evaluations=hit_evaluations,
        success=int(eval_curve[-1] <= cfg.target),
        wall_seconds=elapsed,
        mean_residual=float(np.mean(eval_curve)),
        auc_log_residual=auc_log,
        x0_norm=float(np.linalg.norm(pop[0])),
        curve_eval=eval_curve,
    )


def run_mann(bench: BenchmarkSpec, init: InitSpec, det_cfg: DeterministicConfig, eval_budget: int, seed: int) -> RunResult:
    det_cfg.validate()
    rng = np.random.default_rng(seed)
    x = sample_initial_point(rng, init)
    x0 = x.copy()
    history = []
    t0 = time.perf_counter()
    hit_evaluations = eval_budget + 1

    for n in range(eval_budget):
        tx = benchmark_operator(x, bench)
        alpha_n = det_cfg.alpha(n + 1)
        x = geodesic_interp(x, tx, alpha_n)
        res_n = residual(x, bench)
        history.append(res_n)
        if hit_evaluations == eval_budget + 1 and res_n <= det_cfg.target:
            hit_evaluations = n + 1

    elapsed = time.perf_counter() - t0
    eval_curve = np.array(history, dtype=float)
    auc_log = float(np.trapezoid(np.log10(np.maximum(eval_curve, 1e-14)), dx=1.0))
    return RunResult(
        family='benchmark_suite',
        benchmark=bench.name,
        method='Mann',
        label=f'{bench.name}|Mann',
        seed=seed,
        final_residual=float(eval_curve[-1]),
        hit_evaluations=hit_evaluations,
        success=int(eval_curve[-1] <= det_cfg.target),
        wall_seconds=elapsed,
        mean_residual=float(np.mean(eval_curve)),
        auc_log_residual=auc_log,
        x0_norm=float(np.linalg.norm(x0)),
        curve_eval=eval_curve,
    )


def run_halpern(bench: BenchmarkSpec, init: InitSpec, det_cfg: DeterministicConfig, eval_budget: int, seed: int) -> RunResult:
    det_cfg.validate()
    rng = np.random.default_rng(seed)
    x0 = sample_initial_point(rng, init)
    u = x0.copy()
    x = x0.copy()
    history = []
    t0 = time.perf_counter()
    hit_evaluations = eval_budget + 1

    for n in range(eval_budget):
        tx = benchmark_operator(x, bench)
        alpha_n = det_cfg.alpha(n + 1)
        x = geodesic_interp(u, tx, 1.0 - alpha_n)
        res_n = residual(x, bench)
        history.append(res_n)
        if hit_evaluations == eval_budget + 1 and res_n <= det_cfg.target:
            hit_evaluations = n + 1

    elapsed = time.perf_counter() - t0
    eval_curve = np.array(history, dtype=float)
    auc_log = float(np.trapezoid(np.log10(np.maximum(eval_curve, 1e-14)), dx=1.0))
    return RunResult(
        family='benchmark_suite',
        benchmark=bench.name,
        method='Halpern',
        label=f'{bench.name}|Halpern',
        seed=seed,
        final_residual=float(eval_curve[-1]),
        hit_evaluations=hit_evaluations,
        success=int(eval_curve[-1] <= det_cfg.target),
        wall_seconds=elapsed,
        mean_residual=float(np.mean(eval_curve)),
        auc_log_residual=auc_log,
        x0_norm=float(np.linalg.norm(x0)),
        curve_eval=eval_curve,
    )


def run_scenario(scenario: Scenario, seed: int) -> RunResult:
    if scenario.ga_config is not None:
        return run_ga(scenario.benchmark_spec, scenario.init_spec, scenario.ga_config, seed)
    if scenario.det_config is None:
        raise ValueError('Scenario must carry either a GAConfig or a DeterministicConfig.')

    # Match the deterministic evaluation budget to the GA budget.
    matched_ga_budget = DEFAULT_GA_CONFIG.eval_budget
    if scenario.method == 'Mann':
        return run_mann(scenario.benchmark_spec, scenario.init_spec, scenario.det_config, matched_ga_budget, seed)
    if scenario.method == 'Halpern':
        return run_halpern(scenario.benchmark_spec, scenario.init_spec, scenario.det_config, matched_ga_budget, seed)
    raise ValueError(f'Unknown method {scenario.method!r}.')


# ============================================================
# Statistics
# ============================================================

def bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float], rng: np.random.Generator,
                 n_boot: int = 4000, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        stats[i] = statistic(sample)
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi


def paired_permutation_pvalue(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                              n_perm: int = 20000) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError('paired_permutation_pvalue requires matched lengths.')
    diff = x - y
    observed = abs(float(np.mean(diff)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diff)))
    perm_stats = np.abs(np.mean(signs * diff[None, :], axis=1))
    return float((np.sum(perm_stats >= observed) + 1) / (n_perm + 1))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)
    return float(total / (len(x) * len(y)))


# ============================================================
# Plotting
# ============================================================

def save_method_decay_plot(curves: Dict[str, np.ndarray], outpath: Path, title: str) -> None:
    plt.figure(figsize=(8.2, 5.1), dpi=220)
    x = np.arange(1, next(iter(curves.values())).shape[1] + 1)
    for label, arr in curves.items():
        med = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        plt.plot(x, med, linewidth=2.0, label=label)
        plt.fill_between(x, q25, q75, alpha=0.18)
    plt.yscale('log')
    plt.xlabel('Evaluations of $T$')
    plt.ylabel('Residual $d(x,T x)$')
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def save_ecdf_plot(values: Dict[str, np.ndarray], outpath: Path, title: str) -> None:
    plt.figure(figsize=(8.2, 5.1), dpi=220)
    for label, arr in values.items():
        x = np.sort(np.asarray(arr, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', linewidth=2.0, label=label)
    plt.xscale('log')
    plt.xlabel('Final residual')
    plt.ylabel('ECDF')
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def save_hitting_ecdf(values: Dict[str, np.ndarray], outpath: Path, title: str) -> None:
    plt.figure(figsize=(8.2, 5.1), dpi=220)
    for label, arr in values.items():
        x = np.sort(np.asarray(arr, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', linewidth=2.0, label=label)
    plt.xlabel('Hitting evaluations')
    plt.ylabel('ECDF')
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


# ============================================================
# Scenario design
# ============================================================

DEFAULT_INIT_SPEC = InitSpec(
    rho_work=0.97,
    init_rmin_factor=0.88,
    init_angle_min=-math.pi / 8.0,
    init_angle_max=math.pi / 8.0,
)
DEFAULT_GA_CONFIG = GAConfig(
    pop_size=100,
    generations=250,
    mutation_prob=0.40,
    refresh_eps=0.10,
    refresh_law='uniform',
    tournament_size=2,
    target=1e-5,
)
DEFAULT_DET_CONFIG = DeterministicConfig(target=1e-5, alpha_mode='harmonic')


def build_scenarios(zstar_x: float, zstar_y: float) -> List[Scenario]:
    shifted_norm = math.hypot(zstar_x, zstar_y)
    if shifted_norm >= 0.75:
        raise ValueError('For stability of the working benchmark, choose ||zstar|| < 0.75.')

    benchmarks = [
        BenchmarkSpec(name='central', gamma=0.90, zstar_x=0.0, zstar_y=0.0),
        BenchmarkSpec(name='shifted', gamma=0.90, zstar_x=zstar_x, zstar_y=zstar_y),
    ]

    scenarios: List[Scenario] = []
    for bench in benchmarks:
        scenarios.append(
            Scenario(
                family='benchmark_suite',
                benchmark=bench.name,
                method='GA_0.00',
                label=f'{bench.name}|GA_0.00',
                benchmark_spec=bench,
                init_spec=DEFAULT_INIT_SPEC,
                ga_config=GAConfig(**{**asdict(DEFAULT_GA_CONFIG), 'refresh_eps': 0.00}),
            )
        )
        scenarios.append(
            Scenario(
                family='benchmark_suite',
                benchmark=bench.name,
                method='GA_0.10',
                label=f'{bench.name}|GA_0.10',
                benchmark_spec=bench,
                init_spec=DEFAULT_INIT_SPEC,
                ga_config=GAConfig(**{**asdict(DEFAULT_GA_CONFIG), 'refresh_eps': 0.10}),
            )
        )
        scenarios.append(
            Scenario(
                family='benchmark_suite',
                benchmark=bench.name,
                method='Mann',
                label=f'{bench.name}|Mann',
                benchmark_spec=bench,
                init_spec=DEFAULT_INIT_SPEC,
                det_config=DEFAULT_DET_CONFIG,
            )
        )
        scenarios.append(
            Scenario(
                family='benchmark_suite',
                benchmark=bench.name,
                method='Halpern',
                label=f'{bench.name}|Halpern',
                benchmark_spec=bench,
                init_spec=DEFAULT_INIT_SPEC,
                det_config=DEFAULT_DET_CONFIG,
            )
        )
    return scenarios


# ============================================================
# Orchestration
# ============================================================

def execute_scenarios(scenarios: List[Scenario], runs_per_scenario: int, n_jobs: int) -> List[RunResult]:
    seeds = list(range(1, runs_per_scenario + 1))
    tasks = [(scenario, seed) for scenario in scenarios for seed in seeds]
    results: List[RunResult] = []
    total_tasks = len(tasks)

    if n_jobs == 1:
        for scenario, seed in tqdm(tasks, total=total_tasks, desc='Running simulations'):
            results.append(run_scenario(scenario, seed))
        return results

    max_workers = os.cpu_count() if n_jobs <= 0 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=None) as ex:
        futures = [ex.submit(run_scenario, scenario, seed) for scenario, seed in tasks]
        for fut in tqdm(as_completed(futures), total=total_tasks, desc='Running simulations'):
            results.append(fut.result())
    return results


def run_suite(outdir: Path, runs_per_scenario: int, n_jobs: int, zstar_x: float, zstar_y: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'data').mkdir(exist_ok=True)
    (outdir / 'figures').mkdir(exist_ok=True)

    scenarios = build_scenarios(zstar_x=zstar_x, zstar_y=zstar_y)
    results = execute_scenarios(scenarios, runs_per_scenario=runs_per_scenario, n_jobs=n_jobs)

    rows = []
    curves_by_key: Dict[Tuple[str, str], List[np.ndarray]] = {}
    manifest = []
    for s in scenarios:
        item = {
            'family': s.family,
            'benchmark': s.benchmark,
            'method': s.method,
            'label': s.label,
            'gamma': s.benchmark_spec.gamma,
            'zstar_x': s.benchmark_spec.zstar_x,
            'zstar_y': s.benchmark_spec.zstar_y,
            'rho_work': s.init_spec.rho_work,
            'init_rmin_factor': s.init_spec.init_rmin_factor,
            'init_angle_min': s.init_spec.init_angle_min,
            'init_angle_max': s.init_spec.init_angle_max,
        }
        if s.ga_config is not None:
            item.update({f'ga_{k}': v for k, v in asdict(s.ga_config).items()})
        if s.det_config is not None:
            item.update({f'det_{k}': v for k, v in asdict(s.det_config).items()})
        manifest.append(item)

    for r in results:
        curves_by_key.setdefault((r.benchmark, r.method), []).append(r.curve_eval)
        rows.append({
            'family': r.family,
            'benchmark': r.benchmark,
            'method': r.method,
            'label': r.label,
            'seed': r.seed,
            'final_residual': r.final_residual,
            'hit_evaluations': r.hit_evaluations,
            'success': r.success,
            'wall_seconds': r.wall_seconds,
            'mean_residual': r.mean_residual,
            'auc_log_residual': r.auc_log_residual,
            'x0_norm': r.x0_norm,
        })

    runs_df = pd.DataFrame(rows).sort_values(['benchmark', 'method', 'seed'])
    runs_df.to_csv(outdir / 'data' / 'all_runs.csv', index=False)
    pd.DataFrame(manifest).sort_values(['benchmark', 'method']).to_csv(outdir / 'data' / 'scenario_manifest.csv', index=False)

    rng = np.random.default_rng(20260329)
    summary_rows = []
    for (benchmark, method), grp in runs_df.groupby(['benchmark', 'method'], sort=True):
        finals = grp['final_residual'].to_numpy(dtype=float)
        hits = grp['hit_evaluations'].to_numpy(dtype=float)
        succ = grp['success'].to_numpy(dtype=float)
        walls = grp['wall_seconds'].to_numpy(dtype=float)
        aucs = grp['auc_log_residual'].to_numpy(dtype=float)
        curves = np.vstack(curves_by_key[(benchmark, method)])

        mean_lo, mean_hi = bootstrap_ci(finals, np.mean, rng)
        med_lo, med_hi = bootstrap_ci(finals, np.median, rng)
        succ_lo, succ_hi = bootstrap_ci(succ, np.mean, rng)

        finite_hits = hits[hits <= DEFAULT_GA_CONFIG.eval_budget]
        median_hit = float(np.median(finite_hits)) if len(finite_hits) else math.inf

        summary_rows.append({
            'benchmark': benchmark,
            'method': method,
            'runs': len(grp),
            'mean_final_residual': float(np.mean(finals)),
            'mean_final_ci_lo': mean_lo,
            'mean_final_ci_hi': mean_hi,
            'median_final_residual': float(np.median(finals)),
            'median_final_ci_lo': med_lo,
            'median_final_ci_hi': med_hi,
            'success_rate': float(np.mean(succ)),
            'success_ci_lo': succ_lo,
            'success_ci_hi': succ_hi,
            'median_hit_evaluations': median_hit,
            'mean_wall_seconds': float(np.mean(walls)),
            'median_auc_log_residual': float(np.median(aucs)),
            'median_x0_norm': float(np.median(grp['x0_norm'].to_numpy(dtype=float))),
            'median_curve_last': float(np.median(curves[:, -1])),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(['benchmark', 'method'])
    summary_df.to_csv(outdir / 'data' / 'summary.csv', index=False)

    pair_rows = []
    method_order = ['GA_0.00', 'GA_0.10', 'Mann', 'Halpern']
    for benchmark in sorted(runs_df['benchmark'].unique()):
        sub = runs_df[runs_df['benchmark'] == benchmark].copy()
        baseline = sub[sub['method'] == 'GA_0.00'].sort_values('seed')
        for method in method_order:
            grp = sub[sub['method'] == method].sort_values('seed')
            if method == 'GA_0.00':
                continue
            pval = paired_permutation_pvalue(
                baseline['final_residual'].to_numpy(),
                grp['final_residual'].to_numpy(),
                rng,
            )
            delta = cliffs_delta(grp['final_residual'].to_numpy(), baseline['final_residual'].to_numpy())
            pair_rows.append({
                'benchmark': benchmark,
                'comparison': f'{method} vs GA_0.00',
                'paired_permutation_pvalue_final_residual': pval,
                'cliffs_delta_treatment_vs_baseline': delta,
                'baseline_median_final': float(baseline['final_residual'].median()),
                'treatment_median_final': float(grp['final_residual'].median()),
            })
    pd.DataFrame(pair_rows).sort_values(['benchmark', 'comparison']).to_csv(outdir / 'data' / 'paired_comparisons.csv', index=False)

    # Compact manuscript-ready table
    def fmt_ci(mid: float, lo: float, hi: float) -> str:
        return f'{mid:.3e} [{lo:.3e}, {hi:.3e}]'

    manuscript_rows = []
    for _, row in summary_df.iterrows():
        manuscript_rows.append({
            'Benchmark': row['benchmark'],
            'Method': row['method'],
            'Mean final residual (95% CI)': fmt_ci(row['mean_final_residual'], row['mean_final_ci_lo'], row['mean_final_ci_hi']),
            'Median final residual (95% CI)': fmt_ci(row['median_final_residual'], row['median_final_ci_lo'], row['median_final_ci_hi']),
            'Success rate (95% CI)': f"{row['success_rate']:.2f} [{row['success_ci_lo']:.2f}, {row['success_ci_hi']:.2f}]",
            'Median hitting evaluations': f"{row['median_hit_evaluations']:.0f}" if math.isfinite(row['median_hit_evaluations']) else '>budget',
            'Median AUC log-residual': f"{row['median_auc_log_residual']:.3f}",
        })
    pd.DataFrame(manuscript_rows).to_csv(outdir / 'data' / 'manuscript_table.csv', index=False)

    # Figures benchmark by benchmark
    for benchmark in sorted(runs_df['benchmark'].unique()):
        sub = runs_df[runs_df['benchmark'] == benchmark]
        curves = {
            method: np.vstack(curves_by_key[(benchmark, method)])
            for method in method_order
            if (benchmark, method) in curves_by_key
        }
        save_method_decay_plot(curves, outdir / 'figures' / f'decay_{benchmark}.png', title=f'Convergence profiles on the {benchmark} benchmark')

        finals = {
            method: sub[sub['method'] == method]['final_residual'].to_numpy(dtype=float)
            for method in method_order
        }
        save_ecdf_plot(finals, outdir / 'figures' / f'ecdf_final_{benchmark}.png', title=f'ECDF of final residuals on the {benchmark} benchmark')

        hits = {
            method: sub[sub['method'] == method]['hit_evaluations'].to_numpy(dtype=float)
            for method in method_order
        }
        save_hitting_ecdf(hits, outdir / 'figures' / f'ecdf_hit_{benchmark}.png', title=f'ECDF of hitting evaluations on the {benchmark} benchmark')

    notes = [
        '% Auto-generated notes for the benchmark suite',
        f'% Runs per scenario: {runs_per_scenario}',
        f'% Evaluation budget matched across methods: {DEFAULT_GA_CONFIG.eval_budget}',
        '% Deterministic methods use one evaluation of T per iteration.',
        '% Halpern uses anchor u = x_0.',
        '% Confidence intervals are percentile bootstrap intervals with 4000 resamples.',
        '% Pairwise p-values use paired randomization tests with sign flips against the no-refresh GA baseline (GA_0.00).',
        '% The shifted benchmark is T_{z*}(z) = z* \\oplus (gamma \\otimes R((\\ominus z*) \\oplus z)).',
        f'% Shifted fixed point used here: z* = ({zstar_x:.4f}, {zstar_y:.4f}).',
        '% The Euclidean working ball is a benchmark modelling choice, not a canonical domain associated with the general theory.',
    ]
    (outdir / 'data' / 'manuscript_notes.tex').write_text('\n'.join(notes), encoding='utf-8')


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulation suite for centered and shifted hyperbolic fixed-point benchmarks.')
    parser.add_argument('--outdir', type=Path, default=Path('results'))
    parser.add_argument('--runs', type=int, default=500, help='Independent runs per scenario.')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='Parallel worker processes. Use 1 for serial.')
    parser.add_argument('--zstar-x', type=float, default=0.55, help='x-coordinate of the shifted fixed point.')
    parser.add_argument('--zstar-y', type=float, default=0.25, help='y-coordinate of the shifted fixed point.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_suite(
        outdir=args.outdir,
        runs_per_scenario=args.runs,
        n_jobs=args.jobs,
        zstar_x=args.zstar_x,
        zstar_y=args.zstar_y,
    )


if __name__ == '__main__':
    main()
