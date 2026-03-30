#!/usr/bin/env python
"""
Encodes CIFAR-10 images with two VE SDE models via the probability-flow ODE
and shows their latent codes are highly correlated across architectures.

Usage (fast — load pre-computed codes):
  python exp/exp-latent.py --load_codes figs/latent/codes.npz

Usage (full encode — slow, ~30 min on GPU):
  python exp/exp-latent.py \
      --ckpt_A runs/vesde_cifar10_cont/ckpt/190000 \
      --cfg_A  vesde_ddpm_cont \
      --ckpt_B runs/ncsnpp_cifar10_cont/ckpt/160000 \
      --cfg_B  vesde_ncsnpp_cont \
      --n_images 16 \
      --save_codes figs/latent/codes.npz \
      --output_dir figs/latent/
"""

import sys, os, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import integrate
from scipy.stats import pearsonr
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import get_config
from sde import get_sde
from model import UNet
from score import get_score_fn
from datasets import get_data_scaler, get_dataset


# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon":    False,
    "figure.dpi":        150,
})

COLOR_A   = "#378ADD"   # blue  — Model A
COLOR_B   = "#D85A30"   # coral — Model B
COLOR_SHF = "#888780"   # gray  — shuffled baseline


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg_name: str):
    """
    Load a trained score model from an Orbax checkpoint directory.

    Args:
        ckpt_path: path to the step directory, e.g.
                   'runs/vesde_cifar10_cont/ckpt/190000'
        cfg_name:  one of the keys in config.CONFIGS, e.g. 'vesde_ddpm_cont'

    Returns:
        model:  UNet instance
        params: EMA parameters on the default JAX device
        config: ml_collections.ConfigDict
    """
    config = get_config(cfg_name)
    model  = UNet(config=config)
    path   = str(pathlib.Path(ckpt_path).resolve() / 'default')
    params = ocp.PyTreeCheckpointer().restore(path)['ema_params']
    params = jax.device_put(params, jax.devices()[0])
    return model, params, config


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROBABILITY FLOW ODE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

def make_encoder(model, params, config, rtol: float = 1e-5, atol: float = 1e-5):
    """
    Build an encode(x_np) → z_flat function for a given model.

    The encoder integrates the probability-flow ODE *forward* (t: ε → T),
    mapping a real image x to its uniquely identifiable latent code z = x(T).
    Both the drift direction and the ODE boundary are the same as in
    interpolate.py; only the output (flat numpy array) differs.
    """
    sde, eps = get_sde(config)
    H, C     = config.data.image_size, config.data.num_channels
    shape    = (1, H, H, C)
    scaler   = get_data_scaler(config.data.centered)

    score_fn = get_score_fn(sde, model, params,
                            train=False,
                            continuous=config.training.continuous)

    @jax.jit
    def drift_fn(x, t):
        # Probability-flow ODE drift:  f(x,t) − ½ g(t)² ∇log p_t(x)
        f, _ = sde.reverse_sde(x, t, score_fn(x, t), probability_flow=True)
        return f

    def ode_fn(t, x_flat):
        x_r   = jnp.asarray(x_flat).reshape(shape)
        vec_t = jnp.full((1,), t)
        return np.array(drift_fn(x_r, vec_t)).reshape(-1)

    def encode(x_np: np.ndarray) -> np.ndarray:
        """
        Encode one image to its latent code z = x(T).

        Args:
            x_np: float32 (H, W, C) in [0, 1]
        Returns:
            z_flat: flat numpy array of shape (C*H*W,)
        """
        x_scaled = scaler(x_np)                    # identity for VE SDE (centered=False)
        x_batch  = np.array(x_scaled)[None]        # (1, H, W, C)
        sol = integrate.solve_ivp(
            ode_fn, (eps, sde.T), x_batch.reshape(-1),
            method='RK45', rtol=rtol, atol=atol,
        )
        return sol.y[:, -1]    # z = x(T), shape (C*H*W,)

    return encode


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COLLECT LATENT CODES
# ─────────────────────────────────────────────────────────────────────────────

def collect_codes(
        ckpt_A: str, cfg_A: str,
        ckpt_B: str, cfg_B: str,
        n_images: int = 16,
        rtol: float = 1e-5,
        atol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode `n_images` CIFAR-10 test images with both models.

    Returns:
        codes_A, codes_B: arrays of shape (n_images, D)  where D = C*H*W
    """
    print(f"Loading Model A: {cfg_A}  ←  {ckpt_A}")
    model_A, params_A, config_A = load_model(ckpt_A, cfg_A)
    encode_A = make_encoder(model_A, params_A, config_A, rtol, atol)

    print(f"Loading Model B: {cfg_B}  ←  {ckpt_B}")
    model_B, params_B, config_B = load_model(ckpt_B, cfg_B)
    encode_B = make_encoder(model_B, params_B, config_B, rtol, atol)

    # Load the first n_images from the CIFAR-10 test set (unbatched, no shuffle)
    # Uses config_A for data loading; both models share the same dataset
    print("Loading CIFAR-10 test images …")
    _, eval_ds = get_dataset(config_A)
    imgs_np = np.stack(list(
        eval_ds.unbatch().take(n_images).as_numpy_iterator()
    ))  # (n_images, H, W, C) float32 in [0, 1]

    codes_A, codes_B = [], []
    for idx in range(n_images):
        img = imgs_np[idx]
        zA = encode_A(img)
        zB = encode_B(img)
        codes_A.append(zA)
        codes_B.append(zB)
        print(f"  encoded image {idx + 1}/{n_images}  "
              f"(D={zA.shape[0]})")

    return np.stack(codes_A), np.stack(codes_B)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fig7_dimension_overlay(codes_A: np.ndarray, codes_B: np.ndarray,
                            image_idx: int = 0,
                            n_dims: int = 100,
                            save_path: str | None = None):
    """
    Fig 7: overlay the first `n_dims` dimensions of both latent codes
    for a single image, to show they track each other closely.
    """
    zA = codes_A[image_idx, :n_dims]
    zB = codes_B[image_idx, :n_dims]
    r, _ = pearsonr(zA, zB)
    dims = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(dims, zA, color=COLOR_A, lw=1.2, label="Model A", zorder=3)
    ax.plot(dims, zB, color=COLOR_B, lw=1.2, label="Model B",
            linestyle="--", zorder=2)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Latent value")
    ax.set_xlim(0, n_dims - 1)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Latent code comparison (first {n_dims} dims, $r={r:.3f}$)",
        fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def fig8a_difference_histogram(codes_A: np.ndarray, codes_B: np.ndarray,
                                save_path: str | None = None):
    """
    Fig 8 (left): box plot of |z_A − z_B| per dimension across all images,
    compared to a shuffled baseline (random pairing of dimensions).
    """
    diffs_real = np.abs(codes_A - codes_B).flatten()

    rng = np.random.default_rng(seed=0)
    codes_B_shuf = np.stack([rng.permutation(row) for row in codes_B])
    diffs_shuf   = np.abs(codes_A - codes_B_shuf).flatten()

    flierprops = dict(marker='D', markersize=2.5, linestyle='none',
                      markerfacecolor='#aaa', markeredgecolor='#aaa', alpha=0.5)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    bp = ax.boxplot(
        [diffs_shuf, diffs_real],
        tick_labels=["Shuffled baseline", "Model A vs B"],
        patch_artist=True, widths=0.28,
        vert=False, flierprops=flierprops,
        medianprops=dict(color='black'),
    )
    bp['boxes'][0].set_facecolor("#AAC4FF"); bp['boxes'][0].set_alpha(0.9)   # pastel blue
    bp['boxes'][1].set_facecolor("#FFAAAA"); bp['boxes'][1].set_alpha(0.9)   # pastel red
    ax.set_xlabel("$|z^A_i - z^B_i|$")
    ax.set_title("Dimension-wise encoding differences", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def fig8b_correlation_histogram(codes_A: np.ndarray, codes_B: np.ndarray,
                                 dim_idx: int = 0,
                                 save_path: str | None = None):
    """
    Fig 8 (right): histogram of dimension-wise Pearson r, with an inset
    scatter of Model A vs Model B for one fixed dimension across all images.

    Main histogram: r_d = pearsonr(codes_A[:, d], codes_B[:, d]) for d in 0..D-1
    Inset scatter:  codes_A[:, dim_idx] vs codes_B[:, dim_idx] — 16 points,
                    one per image, for the single dimension dim_idx.
    """
    n_images, D = codes_A.shape

    r_per_dim = np.array([
        pearsonr(codes_A[:, d], codes_B[:, d])[0]
        for d in range(D)
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.hist(r_per_dim, bins=50, range=(0, 1), color="#D32F2F", alpha=0.9, rwidth=1/1.2)
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Count")
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_title("Dimension-wise correlation coefficients", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── inset: one fixed dimension, 16 images ────────────────────────────
    ax_inset = ax.inset_axes([0.24, 0.22, 0.32, 0.34])
    zA = codes_A[:, dim_idx]   # (n_images,)
    zB = codes_B[:, dim_idx]   # (n_images,)
    r_img, _ = pearsonr(zA, zB)

    ax_inset.scatter(zA, zB, s=16, color="#C62828", alpha=0.8, linewidths=0)
    lims = [min(zA.min(), zB.min()), max(zA.max(), zB.max())]
    ax_inset.plot(lims, lims, "k--", lw=0.8)
    ax_inset.set_xlabel("Model A", fontsize=7)
    ax_inset.set_ylabel("Model B", fontsize=7)
    ax_inset.set_title(f"$x_1(T)$", fontsize=7)
    ax_inset.text(0.97, 0.05, f"$r={r_img:.2f}$", transform=ax_inset.transAxes,
                  fontsize=7, ha="right", va="bottom")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def all_three_panels(codes_A: np.ndarray, codes_B: np.ndarray,
                     image_idx: int = 0,
                     save_path: str | None = None):
    """
    Draw all three panels in a single figure (matches paper layout).
    """
    n_images, D = codes_A.shape

    zA_single = codes_A[image_idx, :100]
    zB_single = codes_B[image_idx, :100]
    r_single, _ = pearsonr(zA_single, zB_single)

    diffs_real = np.abs(codes_A - codes_B).flatten()
    rng = np.random.default_rng(seed=0)
    codes_B_shuf = np.stack([rng.permutation(row) for row in codes_B])
    diffs_shuf   = np.abs(codes_A - codes_B_shuf).flatten()

    # Dimension-wise r: D values, one per latent dimension
    r_per_dim = np.array([
        pearsonr(codes_A[:, d], codes_B[:, d])[0]
        for d in range(D)
    ])
    mean_r = r_per_dim.mean()

    fig = plt.figure(figsize=(13, 3.6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── panel 1: dimension overlay ────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    dims = np.arange(100)
    ax1.plot(dims, zA_single, color=COLOR_A, lw=1.2, label="Model A")
    ax1.plot(dims, zB_single, color=COLOR_B, lw=1.2,
             linestyle="--", label="Model B")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Latent value")
    ax1.set_xlim(0, 99)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Latent code comparison (first 100 dims)$)",
                  fontsize=10)
    print(f"Single-image r (first 100 dims): {r_single:.3f}")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── panel 2: difference histogram ────────────────────────────────────
    ax2      = fig.add_subplot(gs[1])
    max_val  = max(diffs_real.max(), diffs_shuf.max())
    bins_diff = np.linspace(0, max_val, 41)
    ax2.hist(diffs_real, bins=bins_diff, color=COLOR_A,
             alpha=0.75, label="Model A vs B")
    ax2.hist(diffs_shuf, bins=bins_diff, color=COLOR_SHF,
             alpha=0.55, label="Shuffled")
    ax2.set_xlabel("$|z^A_i - z^B_i|$")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)
    ax2.set_title("Dimension-wise encoding differences", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── panel 3: dimension-wise correlation histogram + inset scatter ─────
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(r_per_dim, bins=50, color=COLOR_B, alpha=0.80)
    ax3.axvline(mean_r, color=COLOR_A, lw=1.5, linestyle="--",
                label=f"$\\bar{{r}}={mean_r:.3f}$")
    ax3.set_xlabel("Correlation Coefficient")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=9)
    ax3.set_title("Dimension-wise correlation coefficients", fontsize=10)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Inset scatter: all D dims of one image
    ax_inset = ax3.inset_axes([0.05, 0.45, 0.42, 0.50])
    zA_img = codes_A[image_idx]
    zB_img = codes_B[image_idx]
    r_img, _ = pearsonr(zA_img, zB_img)
    ax_inset.scatter(zA_img, zB_img, s=2, color="#C62828",
                     alpha=0.35, linewidths=0)
    lims = [min(zA_img.min(), zB_img.min()),
            max(zA_img.max(), zB_img.max())]
    ax_inset.plot(lims, lims, "k--", lw=0.8)
    ax_inset.set_xlabel("Model A", fontsize=7)
    ax_inset.set_ylabel("Model B", fontsize=7)
    ax_inset.set_title(f"$x_1(T)$  $r={r_img:.2f}$", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Latent code similarity plots (Figs 7 & 8, Song et al. 2021)")
    p.add_argument("--ckpt_A",     default="runs/vesde_cifar10_cont/ckpt/190000",
                   help="Orbax checkpoint step-dir for Model A")
    p.add_argument("--cfg_A",      default="vesde_ddpm_cont",
                   help="Config name for Model A (see config.CONFIGS)")
    p.add_argument("--ckpt_B",     default="runs/ncsnpp_cifar10_cont/ckpt/160000",
                   help="Orbax checkpoint step-dir for Model B")
    p.add_argument("--cfg_B",      default="vesde_ncsnpp_cont",
                   help="Config name for Model B (see config.CONFIGS)")
    p.add_argument("--n_images",   type=int, default=16,
                   help="Number of CIFAR-10 test images to encode")
    p.add_argument("--image_idx",  type=int, default=0,
                   help="Which image to use for the Fig 7 overlay panel")
    p.add_argument("--output_dir", default="figs/latent",
                   help="Directory to save output figures")
    p.add_argument("--load_codes", default=None,
                   help=".npz file with pre-computed codes (skips encoding)")
    p.add_argument("--save_codes", default=None,
                   help="Save encoded codes to .npz for later reuse")
    p.add_argument("--rtol",       type=float, default=1e-5,
                   help="ODE solver relative tolerance")
    p.add_argument("--atol",       type=float, default=1e-5,
                   help="ODE solver absolute tolerance")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load or compute latent codes ──────────────────────────────────────
    if args.load_codes and os.path.isfile(args.load_codes):
        print(f"Loading pre-computed codes from {args.load_codes}")
        data = np.load(args.load_codes)
        codes_A, codes_B = data["codes_A"], data["codes_B"]
    else:
        print("Encoding images with both models …")
        codes_A, codes_B = collect_codes(
            args.ckpt_A, args.cfg_A,
            args.ckpt_B, args.cfg_B,
            n_images=args.n_images,
            rtol=args.rtol,
            atol=args.atol,
        )
        if args.save_codes:
            np.savez(args.save_codes, codes_A=codes_A, codes_B=codes_B)
            print(f"Codes saved to {args.save_codes}")

    print(f"codes_A shape: {codes_A.shape}")   # (n_images, D)
    print(f"codes_B shape: {codes_B.shape}")

    # ── draw plots ────────────────────────────────────────────────────────
    out = args.output_dir
    fig7_dimension_overlay(
        codes_A, codes_B,
        image_idx=args.image_idx,
        save_path=os.path.join(out, "fig7_dim_overlay.png"))
    fig8a_difference_histogram(
        codes_A, codes_B,
        save_path=os.path.join(out, "fig8a_diff_hist.png"))
    fig8b_correlation_histogram(
        codes_A, codes_B,
        save_path=os.path.join(out, "fig8b_corr_hist.png"))


if __name__ == "__main__":
    main()