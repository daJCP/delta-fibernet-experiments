import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SWEEP_PRESETS = {
    "smoke": {
        "densities": [256*2],
        "lambda_tvas": [1e-9],
        "learning_rates": [1e-3],
        "models": ["alpha", "q_alpha", "q_direct"],
        "n_iter": 30000,
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
from run_high_density_q import MODELS, experiment_dir, high_density_config




DEFAULT_PRESET = "focused"
DEFAULT_MAX_WORKERS = 2
DEFAULT_RUN_LABEL = ""


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def result_done(args, model_name):
    cfg = high_density_config(args.n_iter, args.lambda_tva, args.density, args.learning_rate)
    run_args = argparse.Namespace(
        density=args.density,
        lambda_tva=args.lambda_tva,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        run_label=args.run_label,
    )
    return (experiment_dir(run_args, cfg) / model_name / "metrics.json").exists()


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


def run_job(job, render_figures=False, force=False):
    density, lambda_tva, learning_rate, n_iter, model_name, run_label = job
    job_args = argparse.Namespace(
        density=density,
        lambda_tva=lambda_tva,
        learning_rate=learning_rate,
        n_iter=n_iter,
        run_label=run_label,
    )
    if result_done(job_args, model_name) and not force:
        return f"SKIP {model_name} density={density} lambda_tva={lambda_tva:g} lr={learning_rate:g}"

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
        "--render-figures"
    ]
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
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--render-figures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = SWEEP_PRESETS[args.preset]
    densities = parse_int_list(args.densities) if args.densities else preset["densities"]
    lambda_tvas = parse_float_list(args.lambda_tvas) if args.lambda_tvas else preset["lambda_tvas"]
    learning_rates = parse_float_list(args.learning_rates) if args.learning_rates else preset["learning_rates"]
    models = args.models if args.models else preset["models"]
    n_iter = args.n_iter if args.n_iter is not None else preset["n_iter"]

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
            [n_iter],
            models,
            [args.run_label],
        )
    )

    if args.dry_run:
        print(
            f"preset={args.preset}, jobs={len(jobs)}, densities={densities}, "
            f"lambda_tvas={lambda_tvas}, learning_rates={learning_rates}, "
            f"models={models}, n_iter={n_iter}"
        )
        for job in jobs:
            density, lambda_tva, learning_rate, n_iter, model_name, run_label = job
            print(
                f"Would run model={model_name} density={density} "
                f"lambda_tva={lambda_tva:g} lr={learning_rate:g} n_iter={n_iter} "
                f"run_label={run_label or '<none>'}"
            )
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_job, job, args.render_figures, args.force)
            for job in jobs
        ]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
