import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Fibernet2.Generator import Generator
from Fibernet2.generators.DataLoaders3D import DataDeltaLoader, DataLoader


def high_density_config(
    n_iter: int,
    lambda_tva: float,
    density: int,
    learning_rate: float = 1e-3,
) -> dict:
    return {
        "geometry_file": "example/LA_model.vtk",
        "create_TA_maps": True,
        "maps_files": [],
        "geometry_file_smooth": None,
        "area_multiplier": None,
        "maps": 5,
        "density": density,
        "noise_ms": 0,
        "input_precalculated": None,
        "N_eig_max": 200,
        "n_eigs": 20,
        "extend_n_layers": 3,
        "ensemble_size": 10,
        "layers": [20] * 7 + [1],
        "CVlayers": [20] * 5 + [3],
        "lambda_prior": 1e-2,
        "scaled": True,
        "CVmax": 3.0,
        "batch_size": 32,
        "lambda_pde": 1e-4,
        "lambda_tve": 1e-5,
        "lambda_tva": lambda_tva,
        "lambda_df": 1.0,
        "learning_rate": learning_rate,
        "init_key": 0,
        "seed": 456,
        "gen_key_1": 1651,
        "gen_key_2": 1011,
        "nested_density_sampling": True,
        "type_model": "original",
        "n_iter": n_iter,
}


def resolve_config_paths(cfg: dict) -> dict:
    resolved = cfg.copy()
    for key in ("geometry_file", "geometry_file_smooth", "input_precalculated"):
        value = resolved.get(key)
        if value and not Path(value).is_absolute():
            resolved[key] = str(ROOT / value)
    resolved["maps_files"] = [
        str(ROOT / value) if value and not Path(value).is_absolute() else value
        for value in resolved.get("maps_files", [])
    ]
    return resolved


def default_cache_path(density: int) -> Path:
    return EXP_DIR / "data" / f"best_5_maps_density{density}.npz"


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_high_density_data(cfg: dict, cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating high-density synthetic data with density={cfg['density']}")
    generator_cfg = resolve_config_paths(cfg)
    gen = Generator(generator_cfg)
    gen.sampling_TA(cfg)
    saved_params = cfg.copy()
    saved_params["area_multiplier"] = gen.params["area_multiplier"]
    saved_params["maps"] = gen.params["maps"]

    np.savez_compressed(
        cache_path,
        points=np.asarray(gen.points),
        triangs=np.asarray(gen.triangs),
        P_p=np.asarray(gen.P_p),
        P_p_predict=np.asarray(gen.P_p_predict),
        smooth_basis=np.asarray(gen.smooth_basis),
        phis=np.asarray(gen.phis),
        D=np.asarray(gen.D),
        D_n=np.asarray(gen.D_n),
        area=np.asarray(gen.area),
        operator=np.asarray(gen.operator),
        X_e=np.stack([np.asarray(x) for x in gen.X_e]),
        T_e=np.stack([np.asarray(t) for t in gen.T_e]),
        params_json=np.asarray(json.dumps(saved_params, default=json_default)),
    )
    print(f"Saved data cache to {cache_path}")
    return cache_path


def load_high_density_data(cache_path: Path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Data cache does not exist: {cache_path}")

    data = np.load(cache_path, allow_pickle=False)
    gen = Generator.__new__(Generator)
    gen.params = json.loads(data["params_json"].item())
    gen.params["create_TA_maps"] = False
    gen.init_datasets = {"original": False, "delta": False}
    gen.datasets = {"original": DataLoader, "delta": DataDeltaLoader}
    gen.fun_init_data = {"original": gen._init_cartesian, "delta": gen._init_delta}
    gen.init_sampling = True
    gen.init_mesh = True

    gen.points = data["points"]
    gen.triangs = data["triangs"]
    gen.P_p = data["P_p"]
    gen.P_p_predict = data["P_p_predict"]
    gen.smooth_basis = data["smooth_basis"]
    gen.phis = data["phis"]
    gen.D = data["D"]
    gen.D_n = data["D_n"]
    gen.area = float(data["area"].item())
    gen.operator = data["operator"]
    gen.X_e = [x for x in data["X_e"]]
    gen.T_e = [t for t in data["T_e"]]
    gen.kdtree_X = cKDTree(gen.points)
    return gen


def verify_nested_density_cache(low_cache_path: Path, high_cache_path: Path) -> bool:
    low = np.load(low_cache_path, allow_pickle=False)
    high = np.load(high_cache_path, allow_pickle=False)
    low_x = low["X_e"]
    high_x = high["X_e"]

    if low_x.shape[0] != high_x.shape[0]:
        raise ValueError(f"Map count differs: {low_x.shape[0]} vs {high_x.shape[0]}")
    if low_x.shape[2] != high_x.shape[2]:
        raise ValueError(f"Point dimension differs: {low_x.shape[2]} vs {high_x.shape[2]}")
    if low_x.shape[1] > high_x.shape[1]:
        raise ValueError(f"Low cache has more points per map: {low_x.shape[1]} > {high_x.shape[1]}")

    contained_ok = True
    exact_prefix_ok = True
    for map_id in range(low_x.shape[0]):
        low_prefix = low_x[map_id]
        high_prefix = high_x[map_id, : low_x.shape[1]]
        exact_prefix = np.array_equal(low_prefix, high_prefix)
        high_points = {tuple(point.tolist()) for point in high_x[map_id]}
        contained = all(tuple(point.tolist()) in high_points for point in low_prefix)
        print(
            f"map[{map_id}]: low_points={low_x.shape[1]}, high_points={high_x.shape[1]}, "
            f"exact_prefix={exact_prefix}, contained={contained}"
        )
        exact_prefix_ok = exact_prefix_ok and exact_prefix
        contained_ok = contained_ok and contained

    print(f"nested_density_contained_ok={contained_ok}")
    print(f"nested_density_exact_prefix_ok={exact_prefix_ok}")
    return contained_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=30000)
    parser.add_argument("--lambda-tva", type=float, default=1e-9)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--density", type=int, default=24)
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--verify-nested-low", type=Path, default=None)
    parser.add_argument("--verify-nested-high", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_nested_low or args.verify_nested_high:
        if not args.verify_nested_low or not args.verify_nested_high:
            raise ValueError("Use both --verify-nested-low and --verify-nested-high")
        verify_nested_density_cache(args.verify_nested_low, args.verify_nested_high)
        return

    cfg = high_density_config(args.n_iter, args.lambda_tva, args.density, args.learning_rate)
    cache_path = args.data_cache or default_cache_path(args.density)
    save_high_density_data(cfg, cache_path)


if __name__ == "__main__":
    main()
