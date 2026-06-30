import argparse
import itertools
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SWEEP_PRESETS = {
    "smoke": {
        "densities": [256*2],
        "lambda_tvas": [1e-9],
        "learning_rates": [1e-3],
        "models": ["alpha", "q_alpha", "q_direct"],
        "n_iter": [100, 1000],
    },
    "focused": {
        "densities": [16, 24],
        "lambda_tvas": [0.0, 1e-10, 1e-9, 1e-8],
        "learning_rates": [1e-3, 5e-4],
        "models": ["alpha", "q_alpha", "q_direct"],
        "n_iter": 30000,
    },
    "broad": {
        "densities": [8, 16, 24, 32],
        "lambda_tvas": [0.0, 1e-11, 1e-9, 1e-6, 1e-5],
        "learning_rates": [1e-3, 5e-4, 1e-4],
        "models": ["alpha", "q_alpha", "q_direct"],
        "n_iter": 30000,
    },
}

from prepare_high_density_data import EXP_DIR, default_cache_path
from run_high_density_q import MODELS, experiment_dir, high_density_config, slug_value




DEFAULT_PRESET = "focused"
DEFAULT_MAX_WORKERS = 2
DEFAULT_RUN_LABEL = ""


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def resolve_n_iters(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, str):
        return parse_int_list(value)
    return [int(value)]


def result_done(args, model_name):
    return metrics_done(args, model_name) or checkpoint_done(args, model_name)


def model_result_dir(args, model_name):
    cfg = high_density_config(args.n_iter, args.lambda_tva, args.density, args.learning_rate)
    run_args = argparse.Namespace(
        density=args.density,
        lambda_tva=args.lambda_tva,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        run_label=args.run_label,
    )
    return experiment_dir(run_args, cfg) / model_name


def metrics_done(args, model_name):
    return (model_result_dir(args, model_name) / "metrics.json").exists()


def checkpoint_done(args, model_name):
    checkpoint = best_checkpoint_candidate(args, model_name)
    return checkpoint is not None and checkpoint["completed_iter"] >= int(args.n_iter)


def checkpoint_iter(checkpoint_path):
    checkpoint_path = checkpoint_path
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f">> Ignoring unreadable checkpoint {checkpoint_path}: {exc}")
        return None

    return int(payload.get("completed_iter", 0))


def compatible_checkpoint_candidates(args, model_name):
    cfg = high_density_config(args.n_iter, args.lambda_tva, args.density, args.learning_rate)
    root = (
        EXP_DIR
        / "results"
        / f"density{args.density}"
        / f"lambda_tva{slug_value(args.lambda_tva)}"
        / f"lr{slug_value(args.learning_rate)}"
    )
    if not root.exists():
        return []

    candidates = []
    safe_label = args.run_label.replace("/", "_").replace(" ", "_") if args.run_label else ""

    for niter_dir in root.glob("niter*"):
        if not niter_dir.is_dir():
            continue
        try:
            n_iter = int(niter_dir.name.removeprefix("niter"))
        except ValueError:
            continue
        if n_iter > args.n_iter:
            continue

        checkpoint_path = niter_dir / f"batch{cfg['batch_size']}"
        if safe_label:
            checkpoint_path = checkpoint_path / safe_label
        checkpoint_path = checkpoint_path / model_name / "checkpoints" / "latest.pkl"
        completed_iter = checkpoint_iter(checkpoint_path)
        if completed_iter is None:
            continue
        if completed_iter > args.n_iter:
            continue
        candidates.append(
            {
                "path": checkpoint_path,
                "completed_iter": completed_iter,
                "partition_n_iter": n_iter,
            }
        )

    return sorted(
        candidates,
        key=lambda item: (item["completed_iter"], item["partition_n_iter"]),
        reverse=True,
    )


def best_checkpoint_candidate(args, model_name):
    candidates = compatible_checkpoint_candidates(args, model_name)
    return candidates[0] if candidates else None


def partial_checkpoint_exists(args, model_name):
    checkpoint = best_checkpoint_candidate(args, model_name)
    return checkpoint is not None and checkpoint["completed_iter"] < int(args.n_iter)


def prepare_cache(density, force=False):
    cache_path = default_cache_path(density)
    if cache_path.exists() and not force:
        print(f"Data cache already exists: {cache_path}")
        return cache_path

    cmd = [
        sys.executable,
        str(EXP_DIR / "scripts" / "prepare_high_density_data.py"),
        "--density",
        str(density),
    ]
    print("Preparing data cache:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return cache_path


def run_job(
    job,
    render_figures=False,
    force=False,
    log_every=1,
    checkpoint_every=1000,
    no_resume=False,
):
    density, lambda_tva, learning_rate, n_iter, model_name, run_label = job
    job_args = argparse.Namespace(
        density=density,
        lambda_tva=lambda_tva,
        learning_rate=learning_rate,
        n_iter=n_iter,
        run_label=run_label,
    )
    if result_done(job_args, model_name) and not force:
        return (
            f"SKIP {model_name} density={density} lambda_tva={lambda_tva:g} "
            f"lr={learning_rate:g} checkpoint/metrics already complete"
        )

    if partial_checkpoint_exists(job_args, model_name) and not no_resume:
        print(
            f">> Found partial checkpoint for {model_name} density={density} "
            f"lambda_tva={lambda_tva:g} lr={learning_rate:g}; runner will resume it."
        )

    cmd = [
        sys.executable,
        str(EXP_DIR / "scripts" / "run_high_density_q.py"),
        "--model",
        model_name,
        "--density",
        str(density),
        "--lambda-tva",
        f"{lambda_tva:g}",
        "--learning-rate",
        f"{learning_rate:g}",
        "--n-iter",
        str(n_iter),
        "--data-cache",
        str(default_cache_path(density)),
        "--max-arrows",
        "10000", 
        "--render-figures",
        "--log-every",
        str(log_every),
        "--checkpoint-every",
        str(checkpoint_every),
    ]
    if no_resume:
        cmd.append("--no-resume")
    if run_label:
        cmd += ["--run-label", run_label]
    if render_figures:
        cmd.append("--render-figures")

    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return f"DONE {model_name} density={density} lambda_tva={lambda_tva:g} lr={learning_rate:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=SWEEP_PRESETS.keys(), default=DEFAULT_PRESET)
    parser.add_argument("--densities", default=None)
    parser.add_argument("--lambda-tvas", default=None)
    parser.add_argument("--learning-rates", default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), default=None)
    parser.add_argument("--n-iter", "--n-iters", dest="n_iters", default=None)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--render-figures", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = SWEEP_PRESETS[args.preset]
    densities = parse_int_list(args.densities) if args.densities else preset["densities"]
    lambda_tvas = parse_float_list(args.lambda_tvas) if args.lambda_tvas else preset["lambda_tvas"]
    learning_rates = parse_float_list(args.learning_rates) if args.learning_rates else preset["learning_rates"]
    models = args.models if args.models else preset["models"]
    n_iters = resolve_n_iters(args.n_iters if args.n_iters else preset["n_iter"])

    for density in densities:
        if args.dry_run:
            print(f"Would prepare data cache for density={density}: {default_cache_path(density)}")
        else:
            prepare_cache(density, force=args.force_data)

    jobs = list(
        itertools.product(
            densities,
            lambda_tvas,
            learning_rates,
            n_iters,
            models,
            [args.run_label],
        )
    )

    if args.dry_run:
        print(
            f"preset={args.preset}, jobs={len(jobs)}, densities={densities}, "
            f"lambda_tvas={lambda_tvas}, learning_rates={learning_rates}, "
            f"models={models}, n_iters={n_iters}"
        )
        for job in jobs:
            density, lambda_tva, learning_rate, n_iter, model_name, run_label = job
            job_args = argparse.Namespace(
                density=density,
                lambda_tva=lambda_tva,
                learning_rate=learning_rate,
                n_iter=n_iter,
                run_label=run_label,
            )
            status = "complete-checkpoint" if checkpoint_done(job_args, model_name) else "pending"
            if metrics_done(job_args, model_name):
                status = "metrics"
            elif partial_checkpoint_exists(job_args, model_name):
                status = "partial-checkpoint"
            print(
                f"Would run model={model_name} density={density} "
                f"lambda_tva={lambda_tva:g} lr={learning_rate:g} n_iter={n_iter} "
                f"run_label={run_label or '<none>'} status={status}"
            )
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                run_job,
                job,
                args.render_figures,
                args.force,
                args.log_every,
                args.checkpoint_every,
                args.no_resume,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
