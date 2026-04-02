# MLM4 Advanced Machine Learning

Replicated paper: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).

---

## Train

Specify a config (SDE type + model architecture) and an output directory. Checkpoints are saved every 10k steps and training can be resumed automatically.

```bash
python train.py --config <name> --logdir runs/<name>
```

Available configs: `vpsde_ddpm_cont`, `vpsde_ddpm_disc`, `vesde_ddpm_disc`, `vesde_ddpm_cont`, `subvpsde_ddpm_cont`, `vpsde_ddpmpp_cont`, `vesde_ncsnpp_cont`, `vesde_ncsnpp_celeba_disc`, `vesde_ncsnpp_celeba_cont`

Optional: `--n_iters <int>` to override the number of training steps, `--drive_backup <path>` to mirror checkpoints to a Drive folder after each save.

---

## Checkpoints

Download `runs.zip` from [Google Drive](https://drive.google.com/file/d/1GsP-v-4Mgs8Nfye9RXF5WuJIG4dOBqZe/view?usp=sharing) and unzip at the repo root. Experiment scripts expect checkpoints at `runs/<name>/ckpt/<step>`.

```
runs/
  vpsde_ddpm_cont/ckpt/
  vesde_ddpm_disc/ckpt/
  ...
```

---

## Experiments

**PC sampling**: replicates Table 1 (FID/IS on CIFAR-10). Runs the Predictor-Corrector sampler (Section 4.2) with different predictor/corrector combinations across VE-SDE and VP-SDE models.
```bash
python exp/pc_samplers.py vesde --variant disc
python exp/pc_samplers.py ddpm  --variant cont
```

**ODE probability flow NFE**: replicates Figure 3 (middle). Measures how many score function evaluations the adaptive ODE solver needs to reach a given tolerance.
```bash
python exp/prob_flow_nfe.py
```

**Likelihood evaluation**: replicates Table 2 (NLL on CIFAR-10). Uses the probability flow ODE to convert the SDE into an exact likelihood via the instantaneous change-of-variables formula, with Hutchinson trace estimation for the divergence term.
```bash
python exp/likelihood_eval.py --ckpt runs/<name>/ckpt/<step> --config <name>
```

**Inpainting**: replicates Figure 4 (Inpainting). Masks a region and fills it in by running the reverse SDE only over the masked pixels while keeping the rest controlled at each step.
```bash
python exp/inpaint.py --ckpt runs/<name>/ckpt/<step> --config <name> --mask right_half
```

**Latent interpolation**: replicates Figure 3 (right: latent space interpolation). Encodes two real images to their SDE latent codes via the probability flow ODE, interpolates in that space, then decodes back.
```bash
python exp/interpolate.py --n_interp 8
```

**Uniquely identifiable encoding**: replicates 4.3 paragraph 3 and Figures 7 and 8. Encodes images with two different trained models and checks that the latent codes agree.
```bash
python exp/unique_enc.py --ckpt_A runs/<name>/ckpt/<step> --cfg_A <name>
```

**Evaluate (FID / IS)**: replicates Table 3. Compute FID and IS using tfgan.
```bash
python exp/eval.py
```
