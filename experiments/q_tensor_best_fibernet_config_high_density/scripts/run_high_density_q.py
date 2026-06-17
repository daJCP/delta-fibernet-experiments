import argparse
import json
import sys
from pathlib import Path

import jax as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "q_tensor_report" / "src"))

import jaxpinns.architectures as jxp_ar
import jaxpinns.optimizers as jxp_op
from jaxpinns.loggers import logger

from Fibernet2.FiberNet import FiberNet
from Fibernet2.Generator import Generator
from q_fibernet import QAlphaFiberNet, QDirectFiberNet


MODELS = {
    "alpha": FiberNet,
    "q_alpha": QAlphaFiberNet,
    "q_direct": QDirectFiberNet,
}


def high_density_config(n_iter: int, lambda_tva: float, density: int) -> dict:
    return {
        "geometry_file": str(ROOT / "example" / "LA_model.vtk"),
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
        "learning_rate": 1e-3,
        "init_key": 0,
        "seed": 456,
        "gen_key_1": 1651,
        "gen_key_2": 1011,
        "type_model": "original",
        "n_iter": n_iter,
    }


class ExperimentHandler:
    def __init__(self, model_cls, params, gen):
        self.hiperparams = params.copy()
        self.dataset = gen.dataset(self.hiperparams)
        init_key = jx.random.PRNGKey(self.hiperparams["init_key"])
        self.model = model_cls(
            dataset=self.dataset,
            CVmax=self.hiperparams["CVmax"],
            lambda_pde=self.hiperparams["lambda_pde"],
            lambda_tve=self.hiperparams["lambda_tve"],
            lambda_tva=self.hiperparams["lambda_tva"],
            lambda_df=self.hiperparams["lambda_df"],
        )
        cv_out = 4 if model_cls is QDirectFiberNet else 3
        self.model.architecture(
            jxp_ar.MLP,
            ([3] + [20] * 5 + [cv_out],),
            self.dataset.Tmax.shape[0],
            jxp_ar.MLP,
            ([3] + [20] * 7 + [1],),
            init_key=init_key,
        )
        self.model.optimizer(jxp_op.adam, self.hiperparams["learning_rate"], self.model.loss)
        log_keys = ["loss", "loss_data", "loss_pde", "loss_regu"]
        log_funs = [self.model.loss, self.model.loss_data, self.model.loss_pde, self.model.loss_regu]
        self.model.logger(logger, log_keys, log_keys, log_funs, io_step=100)

    def train(self, epochs):
        self.model.train(self.dataset, nIter=epochs, ntk_weights=False)

    def predict_principal_fibers(self):
        triangs = self.dataset.triangs
        centroids_basis = self.dataset.P_p_predict[triangs].mean(axis=1)
        centroids_x = self.dataset.X[triangs].mean(axis=1)
        vals = []
        vecs = []
        for i in range(24):
            st = int(len(centroids_x) * i / 24)
            ed = int(len(centroids_x) * (i + 1) / 24)
            outs = self.model.predict(centroids_x[st:ed], centroids_basis[st:ed])
            v, w = np.linalg.eigh(np.asarray(jx.device_get(outs[4])))
            vals.append(np.asarray(v))
            vecs.append(np.asarray(w))
        return np.vstack(vals), np.vstack(vecs)[:, :, -1]


def angular_error_deg(pred, truth):
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
    truth = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(pred * truth, axis=1)), 0.0, 1.0)))


def activation_rmse(handler):
    rmses = []
    params = handler.model.get_params(handler.model.opt_state)
    for i, x in enumerate(handler.dataset.X_e):
        y_pred = np.asarray(handler.model.AT_NN(params, x)[..., i]) * handler.model.Tmax[i]
        y_true = np.asarray(handler.dataset.T_e[i]) * handler.model.Tmax[i]
        rmses.append(float(np.sqrt(np.mean((y_pred - y_true) ** 2))))
    return rmses


def to_float(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else float("nan")


def metrics_from_run(handler, truth_fibers):
    _, pred_fibers = handler.predict_principal_fibers()
    err = angular_error_deg(pred_fibers, truth_fibers)
    logs = handler.model.training_log
    reg = logs["loss_regu"][-1]
    reg_orient = reg[1] if isinstance(reg, tuple) else np.asarray(reg)[1]
    rmses = activation_rmse(handler)
    return {
        "angular_error_deg_mean": float(np.mean(err)),
        "angular_error_deg_median": float(np.median(err)),
        "angular_error_deg_p25": float(np.percentile(err, 25)),
        "angular_error_deg_p75": float(np.percentile(err, 75)),
        "angular_error_deg_p95": float(np.percentile(err, 95)),
        "angular_error_deg_max": float(np.max(err)),
        "activation_observed_rmse_by_map": rmses,
        "activation_observed_rmse_mean": float(np.mean(rmses)),
        "final_loss": to_float(logs["loss"][-1]),
        "final_loss_data": to_float(logs["loss_data"][-1]),
        "final_loss_pde": to_float(logs["loss_pde"][-1]),
        "final_loss_regu_orient": to_float(reg_orient),
        "final_loss_regu_orient_weighted": to_float(handler.hiperparams["lambda_tva"] * reg_orient),
        "loss": [to_float(v) for v in logs["loss"]],
        "loss_data": [to_float(v) for v in logs["loss_data"]],
        "loss_pde": [to_float(v) for v in logs["loss_pde"]],
        "loss_regu_orient": [to_float(v[1] if isinstance(v, tuple) else np.asarray(v)[1]) for v in logs["loss_regu"]],
        "angular_errors": err.tolist(),
        "pred_fibers": pred_fibers.tolist(),
        "truth_fibers": truth_fibers.tolist(),
    }


def make_mesh(points, triangs):
    faces = np.hstack([np.full((triangs.shape[0], 1), 3), triangs]).astype(np.int64)
    return pv.PolyData(points, faces)


def apply_view(plotter, view_name):
    if view_name == "isometric":
        plotter.view_isometric()
    elif view_name == "front":
        plotter.view_xy()
    elif view_name == "back":
        plotter.view_xy(negative=True)
    elif view_name == "left":
        plotter.view_yz()
    elif view_name == "right":
        plotter.view_yz(negative=True)
    elif view_name == "top":
        plotter.view_xz()
    plotter.camera.zoom(1.18)


def add_fiber_panel(plotter, mesh, centers, vectors, ids, title, color, mag, view_name):
    plotter.add_mesh(mesh, color="white", opacity=0.23, show_edges=False, smooth_shading=True)
    plotter.add_mesh(mesh.extract_surface(), color="lightgray", opacity=0.16, show_edges=False)
    plotter.add_arrows(centers[ids], vectors[ids], mag=mag, color=color)
    plotter.add_text(title, font_size=15, color="black")
    apply_view(plotter, view_name)


def render_fiber_views(results, gen, out_dir, run_slug, max_arrows):
    pv.OFF_SCREEN = True
    fig_dir = out_dir / "figures" / run_slug
    fig_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(gen.points)
    triangs = np.asarray(gen.triangs)
    centers = points[triangs].mean(axis=1)
    mesh = make_mesh(points, triangs)
    truth = np.asarray(gen.D)[:, :, -1]
    ids = np.arange(0, len(centers), max(1, len(centers) // max_arrows))
    mag = np.linalg.norm(np.asarray(mesh.bounds)[1::2] - np.asarray(mesh.bounds)[::2]) * 0.011
    panels = [("truth", truth, "Ground truth", "black")]
    panels += [(name, np.asarray(metrics["pred_fibers"]), name, color) for name, metrics, color in [
        ("alpha", results["alpha"], "darkorange"),
        ("q_alpha", results["q_alpha"], "seagreen"),
        ("q_direct", results["q_direct"], "crimson"),
    ]]
    views = ["isometric", "front", "left", "right", "top"]
    for view in views:
        plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(2200, 1700))
        plotter.set_background("white")
        for i, (_, vectors, title, color) in enumerate(panels):
            plotter.subplot(i // 2, i % 2)
            add_fiber_panel(plotter, mesh, centers, vectors, ids, title, color, mag, view)
        plotter.screenshot(fig_dir / f"fiber_field_dense_{view}.png")
        plotter.close()

    for name, metrics, color in [("alpha", results["alpha"], "darkorange"), ("q_alpha", results["q_alpha"], "seagreen"), ("q_direct", results["q_direct"], "crimson")]:
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1900, 850))
        plotter.set_background("white")
        for idx, (title, vectors, c) in enumerate([("Ground truth", truth, "black"), (name, np.asarray(metrics["pred_fibers"]), color)]):
            plotter.subplot(0, idx)
            add_fiber_panel(plotter, mesh, centers, vectors, ids, title, c, mag, "isometric")
        plotter.screenshot(fig_dir / f"{name}_dense_isometric.png")
        plotter.close()


def render_error_views(results, gen, out_dir, run_slug):
    pv.OFF_SCREEN = True
    fig_dir = out_dir / "figures" / run_slug
    points = np.asarray(gen.points)
    triangs = np.asarray(gen.triangs)
    base_mesh = make_mesh(points, triangs)
    views = ["isometric", "front", "left", "right", "top"]
    for view in views:
        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2200, 780))
        plotter.set_background("white")
        for i, (name, metrics) in enumerate(results.items()):
            mesh = base_mesh.copy()
            mesh.cell_data["angular_error"] = np.asarray(metrics["angular_errors"], dtype=np.float32)
            plotter.subplot(0, i)
            plotter.add_mesh(
                mesh,
                scalars="angular_error",
                cmap="inferno",
                clim=(0.0, 90.0),
                show_edges=False,
                smooth_shading=True,
                scalar_bar_args={"title": "Error angular (deg)", "vertical": True},
            )
            plotter.add_text(name, font_size=15, color="black")
            apply_view(plotter, view)
        plotter.screenshot(fig_dir / f"angular_error_dense_{view}.png")
        plotter.close()


def render_summary_charts(results, out_dir, run_slug):
    fig_dir = out_dir / "figures" / run_slug
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, metrics in results.items():
        ax.hist(metrics["angular_errors"], bins=45, alpha=0.45, label=name)
    ax.set_xlabel("fiber orientation error [deg]")
    ax.set_ylabel("cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "angular_error_histogram.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results)
    means = [results[n]["angular_error_deg_mean"] for n in names]
    meds = [results[n]["angular_error_deg_median"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.18, means, width=0.36, label="mean")
    ax.bar(x + 0.18, meds, width=0.36, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("error [deg]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "angular_error_bars.png", dpi=220)
    plt.close(fig)


def render_figures(results, gen, out_dir, run_slug, max_arrows):
    render_fiber_views(results, gen, out_dir, run_slug, max_arrows)
    render_error_views(results, gen, out_dir, run_slug)
    render_summary_charts(results, out_dir, run_slug)


def strip_arrays(results):
    skip = {"loss", "loss_data", "loss_pde", "loss_regu_orient", "angular_errors", "pred_fibers", "truth_fibers"}
    return {model: {k: v for k, v in metrics.items() if k not in skip} for model, metrics in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=30000)
    parser.add_argument("--lambda-tva", type=float, default=1e-9)
    parser.add_argument("--density", type=int, default=24)
    parser.add_argument("--max-arrows", type=int, default=2600)
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), default=list(MODELS.keys()))
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    lam_slug = "lambda1e-9" if args.lambda_tva == 1e-9 else f"lambda{args.lambda_tva:g}".replace(".", "p")
    run_slug = f"{lam_slug}_density{args.density}_best5"
    if args.run_label:
        safe_label = args.run_label.replace("/", "_").replace(" ", "_")
        run_slug = f"{run_slug}_{safe_label}"
    out_dir = EXP_DIR / "results" / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "configs").mkdir(parents=True, exist_ok=True)
    cfg = high_density_config(args.n_iter, args.lambda_tva, args.density)
    cfg_name = f"best_5_maps_{lam_slug}_density{args.density}"
    if args.run_label:
        cfg_name = f"{cfg_name}_{safe_label}"
    with open(EXP_DIR / "configs" / f"{cfg_name}.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Preparing shared synthetic data with density={args.density}")
    gen = Generator(cfg)
    truth_fibers = np.asarray(gen.D)[:, :, -1]

    results = {}
    for model_name in args.models:
        model_cls = MODELS[model_name]
        print(f"Training {model_name} with density={args.density}")
        handler = ExperimentHandler(model_cls, cfg, gen)
        handler.train(args.n_iter)
        metrics = metrics_from_run(handler, truth_fibers)
        results[model_name] = metrics
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "metrics.json", "w") as f:
            json.dump({k: v for k, v in metrics.items() if k not in {"pred_fibers", "truth_fibers", "angular_errors", "loss", "loss_data", "loss_pde", "loss_regu_orient"}}, f, indent=2)
        np.savez_compressed(
            model_dir / "reconstruction.npz",
            pred_fibers=np.asarray(metrics["pred_fibers"]),
            truth_fibers=np.asarray(metrics["truth_fibers"]),
            angular_errors=np.asarray(metrics["angular_errors"]),
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(strip_arrays(results), f, indent=2)
    if set(results) == set(MODELS):
        render_figures(results, gen, EXP_DIR, run_slug, args.max_arrows)
    else:
        print("Skipping combined figures until all models are available.")
    print(json.dumps(strip_arrays(results), indent=2))


if __name__ == "__main__":
    main()
