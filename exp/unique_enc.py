"""
python exp/unique_enc.py \
    --ckpt_A runs/vesde_cifar10_cont/ckpt/190000 \
    --cfg_A  vesde_ddpm_cont \
    --ckpt_B runs/ncsnpp_cifar10_cont/ckpt/160000 \
    --cfg_B  vesde_ncsnpp_cont \
    --n_images 16 \
    --save_codes figs/latent/codes.npz \
    --output_dir figs/latent/
    
python exp/unique_enc.py --load_codes figs/latent/codes.npz
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

COLOR_A   = "#378ADD"
COLOR_B   = "#D85A30"
COLOR_SHF = "#888780"


def load_model(ckpt_path: str, cfg_name: str):
    config = get_config(cfg_name)
    model  = UNet(config=config)
    path   = str(pathlib.Path(ckpt_path).resolve() / 'default')
    params = jax.device_put(ocp.PyTreeCheckpointer().restore(path)['ema_params'], jax.devices()[0])
    return model, params, config


def make_encoder(model, params, config, rtol: float = 1e-5, atol: float = 1e-5):
    sde, eps = get_sde(config)
    H, C     = config.data.image_size, config.data.num_channels
    shape    = (1, H, H, C)
    scaler   = get_data_scaler(config.data.centered)
    score_fn = get_score_fn(sde, model, params, train=False, continuous=config.training.continuous)

    @jax.jit
    def drift_fn(x, t):
        f, _ = sde.reverse_sde(x, t, score_fn(x, t), probability_flow=True)
        return f

    def ode_fn(t, x_flat):
        return np.array(drift_fn(jnp.asarray(x_flat).reshape(shape), jnp.full((1,), t))).reshape(-1)

    def encode(x_np: np.ndarray) -> np.ndarray:
        x_batch = np.array(scaler(x_np))[None]
        sol = integrate.solve_ivp(ode_fn, (eps, sde.T), x_batch.reshape(-1), method='RK45', rtol=rtol, atol=atol)
        return sol.y[:, -1]

    return encode


def collect_codes(ckpt_A, cfg_A, ckpt_B, cfg_B, n_images=16, rtol=1e-5, atol=1e-5):
    print(f"Loading Model A: {cfg_A}  ←  {ckpt_A}")
    model_A, params_A, config_A = load_model(ckpt_A, cfg_A)
    encode_A = make_encoder(model_A, params_A, config_A, rtol, atol)

    print(f"Loading Model B: {cfg_B}  ←  {ckpt_B}")
    model_B, params_B, config_B = load_model(ckpt_B, cfg_B)
    encode_B = make_encoder(model_B, params_B, config_B, rtol, atol)

    print("Loading CIFAR-10 test images …")
    _, eval_ds = get_dataset(config_A)
    imgs_np = np.stack(list(eval_ds.unbatch().take(n_images).as_numpy_iterator()))

    codes_A, codes_B = [], []
    for idx in range(n_images):
        img = imgs_np[idx]
        zA, zB = encode_A(img), encode_B(img)
        codes_A.append(zA)
        codes_B.append(zB)
        print(f"  encoded image {idx + 1}/{n_images}  (D={zA.shape[0]})")

    return np.stack(codes_A), np.stack(codes_B)


def fig7_dimension_overlay(codes_A, codes_B, image_idx=0, n_dims=100, save_path=None):
    zA, zB = codes_A[image_idx, :n_dims], codes_B[image_idx, :n_dims]
    dims = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(dims, zA, color=COLOR_A, lw=1.2, label="Model A", zorder=3)
    ax.plot(dims, zB, color=COLOR_B, lw=1.2, linestyle="--", label="Model B", zorder=2)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Latent value")
    ax.set_xlim(0, n_dims - 1)
    ax.legend(loc="lower right")
    ax.set_title(f"Latent code comparison (first {n_dims} dims)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def fig8a_difference_histogram(codes_A, codes_B, save_path=None):
    diffs_real = np.abs(codes_A - codes_B).flatten()
    rng = np.random.default_rng(seed=0)
    diffs_shuf = np.abs(codes_A - np.stack([rng.permutation(row) for row in codes_B])).flatten()

    flierprops = dict(marker='D', markersize=2.5, linestyle='none',
                      markerfacecolor='#aaa', markeredgecolor='#aaa', alpha=0.5)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bp = ax.boxplot(
        [diffs_shuf, diffs_real],
        tick_labels=["Shuffled", "Model A vs B"],
        patch_artist=True, widths=0.28,
        vert=False, flierprops=flierprops,
        medianprops=dict(color='black'),
    )
    bp['boxes'][0].set_facecolor("#AAC4FF"); bp['boxes'][0].set_alpha(0.9)
    bp['boxes'][1].set_facecolor("#FFAAAA"); bp['boxes'][1].set_alpha(0.9)
    ax.set_xlabel("$|z^A_i - z^B_i|$")
    ax.set_title("Dimension-wise encoding differences", fontsize=11)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='grey', alpha=0.3, linewidth=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def fig8b_correlation_histogram(codes_A, codes_B, dim_idx=0, save_path=None):
    n_images, D = codes_A.shape
    r_per_dim = np.array([pearsonr(codes_A[:, d], codes_B[:, d])[0] for d in range(D)])

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(r_per_dim, bins=50, range=(0, 1), color="#D32F2F", alpha=0.9, rwidth=1/1.2)
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Count")
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0, 400, 800, 1200])
    ax.set_title("Dimension-wise correlation coefficients", fontsize=11)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='grey', alpha=0.3, linewidth=0.8)
    ax.yaxis.grid(True, color='grey', alpha=0.3, linewidth=0.8)

    ax_inset = ax.inset_axes([0.22, 0.20, 0.34, 0.48])
    zA, zB = codes_A[:, dim_idx], codes_B[:, dim_idx]
    r_img, _ = pearsonr(zA, zB)
    ax_inset.scatter(zA, zB, s=22, color="#C62828", alpha=0.8, linewidths=0)
    lims = [min(zA.min(), zB.min()), max(zA.max(), zB.max())]
    ax_inset.plot(lims, lims, "k--", lw=0.8)
    ax_inset.set_xlabel("Model A", fontsize=7)
    ax_inset.set_ylabel("Model B", fontsize=7)
    ax_inset.set_title(f"$x_1(T)$", fontsize=7)
    ax_inset.text(0.97, 0.05, f"$r={r_img:.2f}$", transform=ax_inset.transAxes,
                  fontsize=7, ha="right", va="bottom")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#aaa')
        spine.set_alpha(0.6)
        spine.set_linewidth(0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def all_three_panels(codes_A, codes_B, image_idx=0, save_path=None):
    n_images, D = codes_A.shape

    zA_single, zB_single = codes_A[image_idx, :100], codes_B[image_idx, :100]
    r_single, _ = pearsonr(zA_single, zB_single)

    diffs_real = np.abs(codes_A - codes_B).flatten()
    rng = np.random.default_rng(seed=0)
    diffs_shuf = np.abs(codes_A - np.stack([rng.permutation(row) for row in codes_B])).flatten()

    r_per_dim = np.array([pearsonr(codes_A[:, d], codes_B[:, d])[0] for d in range(D)])
    mean_r = r_per_dim.mean()

    fig = plt.figure(figsize=(13, 3.6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1  = fig.add_subplot(gs[0])
    dims = np.arange(100)
    ax1.plot(dims, zA_single, color=COLOR_A, lw=1.2, label="Model A")
    ax1.plot(dims, zB_single, color=COLOR_B, lw=1.2, linestyle="--", label="Model B")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Latent value")
    ax1.set_xlim(0, 99)
    ax1.legend(fontsize=9)
    ax1.set_title("Latent code comparison (first 100 dims)", fontsize=10)
    print(f"Single-image r (first 100 dims): {r_single:.3f}")

    ax2       = fig.add_subplot(gs[1])
    bins_diff = np.linspace(0, max(diffs_real.max(), diffs_shuf.max()), 41)
    ax2.hist(diffs_real, bins=bins_diff, color=COLOR_A,   alpha=0.75, label="Model A vs B")
    ax2.hist(diffs_shuf, bins=bins_diff, color=COLOR_SHF, alpha=0.55, label="Shuffled")
    ax2.set_xlabel("$|z^A_i - z^B_i|$")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)
    ax2.set_title("Dimension-wise encoding differences", fontsize=10)

    ax3 = fig.add_subplot(gs[2])
    ax3.hist(r_per_dim, bins=50, color=COLOR_B, alpha=0.80)
    ax3.axvline(mean_r, color=COLOR_A, lw=1.5, linestyle="--", label=f"$\\bar{{r}}={mean_r:.3f}$")
    ax3.set_xlabel("Correlation Coefficient")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=9)
    ax3.set_title("Dimension-wise correlation coefficients", fontsize=10)

    ax_inset = ax3.inset_axes([0.05, 0.45, 0.42, 0.50])
    zA_img, zB_img = codes_A[image_idx], codes_B[image_idx]
    r_img, _ = pearsonr(zA_img, zB_img)
    ax_inset.scatter(zA_img, zB_img, s=2, color="#C62828", alpha=0.35, linewidths=0)
    lims = [min(zA_img.min(), zB_img.min()), max(zA_img.max(), zB_img.max())]
    ax_inset.plot(lims, lims, "k--", lw=0.8)
    ax_inset.set_xlabel("Model A", fontsize=7)
    ax_inset.set_ylabel("Model B", fontsize=7)
    ax_inset.set_title(f"$x_1(T)$  $r={r_img:.2f}$", fontsize=7)
    ax_inset.tick_params(labelsize=6)

    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_A",     default="runs/vesde_cifar10_cont/ckpt/190000")
    p.add_argument("--cfg_A",      default="vesde_ddpm_cont")
    p.add_argument("--ckpt_B",     default="runs/ncsnpp_cifar10_cont/ckpt/160000")
    p.add_argument("--cfg_B",      default="vesde_ncsnpp_cont")
    p.add_argument("--n_images",   type=int, default=16)
    p.add_argument("--image_idx",  type=int, default=0)
    p.add_argument("--output_dir", default="figs/latent")
    p.add_argument("--load_codes", default=None)
    p.add_argument("--save_codes", default=None)
    p.add_argument("--rtol",       type=float, default=1e-5)
    p.add_argument("--atol",       type=float, default=1e-5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_codes and os.path.isfile(args.load_codes):
        print(f"Loading pre-computed codes from {args.load_codes}")
        data = np.load(args.load_codes)
        codes_A, codes_B = data["codes_A"], data["codes_B"]
    else:
        print("Encoding images with both models …")
        codes_A, codes_B = collect_codes(
            args.ckpt_A, args.cfg_A, args.ckpt_B, args.cfg_B,
            n_images=args.n_images, rtol=args.rtol, atol=args.atol,
        )
        if args.save_codes:
            np.savez(args.save_codes, codes_A=codes_A, codes_B=codes_B)
            print(f"Codes saved to {args.save_codes}")

    print(f"codes_A shape: {codes_A.shape}")
    print(f"codes_B shape: {codes_B.shape}")

    out = args.output_dir
    fig7_dimension_overlay(codes_A, codes_B, image_idx=args.image_idx,
                           save_path=os.path.join(out, "fig7_dim_overlay.png"))
    fig8a_difference_histogram(codes_A, codes_B,
                               save_path=os.path.join(out, "fig8a_diff_hist.png"))
    fig8b_correlation_histogram(codes_A, codes_B,
                                save_path=os.path.join(out, "fig8b_corr_hist.png"))


if __name__ == "__main__":
    main()
