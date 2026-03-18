import ml_collections


def _cifar10_data(centered):
    d = ml_collections.ConfigDict()
    d.dataset = 'CIFAR10'; d.image_size = 32; d.num_channels = 3
    d.centered = centered; d.random_flip = True; d.uniform_dequantization = False
    return d


def _ddpm_model(ema_rate, embedding_type, **sde_params):
    m = ml_collections.ConfigDict()
    m.nf = 128; m.ch_mult = (1, 2, 2, 2); m.num_res_blocks = 2
    m.attn_resolutions = (16,); m.dropout = 0.1
    m.ema_rate = ema_rate
    m.fir = False; m.skip_rescale = False
    m.resblock_type = 'ddpm'; m.progressive_input = 'none'
    m.embedding_type = embedding_type
    m.upsample_method = 'bilinear'
    for k, v in sde_params.items():
        setattr(m, k, v)
    return m


def _training(sde, continuous, reduce_mean):
    t = ml_collections.ConfigDict()
    t.batch_size = 128; t.n_iters = 1_300_000
    t.learning_rate = 2e-4; t.warmup = 5000; t.grad_clip = 1.0
    t.sde = sde; t.sde_N = 1000
    t.continuous = continuous; t.reduce_mean = reduce_mean
    return t


def get_vpsde_ddpm(continuous=True):
    cfg = ml_collections.ConfigDict()
    cfg.data = _cifar10_data(centered=True)
    cfg.model = _ddpm_model(ema_rate=0.9999, embedding_type='positional',
                            beta_min=0.1, beta_max=20.0)
    cfg.training = _training('vpsde', continuous, reduce_mean=True)
    return cfg


def get_vesde_ddpm(continuous=False):
    cfg = ml_collections.ConfigDict()
    cfg.data = _cifar10_data(centered=False)
    cfg.model = _ddpm_model(ema_rate=0.999,
                            embedding_type='fourier' if continuous else 'positional',
                            sigma_min=0.01, sigma_max=50.0)
    cfg.training = _training('vesde', continuous, reduce_mean=False)
    return cfg


def _ncsnpp_model(ema_rate, embedding_type, fir, progressive_input, **sde_params):
    """BigGAN resblocks + skip_rescale=True + 4 res blocks per resolution."""
    m = ml_collections.ConfigDict()
    m.nf = 128; m.ch_mult = (1, 2, 2, 2); m.num_res_blocks = 4
    m.attn_resolutions = (16,); m.dropout = 0.1
    m.ema_rate = ema_rate
    m.fir = fir; m.skip_rescale = True
    m.resblock_type = 'biggan'; m.progressive_input = progressive_input
    m.embedding_type = embedding_type
    m.upsample_method = 'bilinear'
    for k, v in sde_params.items():
        setattr(m, k, v)
    return m


def get_ddpmpp_vpsde():
    """DDPM++ continuous VP SDE.
    Reference: score_sde/configs/vp/cifar10_ddpmpp_continuous.py
    No FIR, no progressive input, positional (sinusoidal) embedding.
    """
    cfg = ml_collections.ConfigDict()
    cfg.data = _cifar10_data(centered=True)
    cfg.model = _ncsnpp_model(ema_rate=0.9999, embedding_type='positional',
                              fir=False, progressive_input='none',
                              beta_min=0.1, beta_max=20.0)
    cfg.training = _training('vpsde', continuous=True, reduce_mean=True)
    return cfg


def get_ncsnpp_vesde():
    """NCSN++ continuous VE SDE.
    Reference: score_sde/configs/ve/cifar10_ncsnpp_continuous.py
    FIR, residual progressive input, Fourier (GFF) embedding.
    """
    cfg = ml_collections.ConfigDict()
    cfg.data = _cifar10_data(centered=False)
    cfg.model = _ncsnpp_model(ema_rate=0.999, embedding_type='fourier',
                              fir=True, progressive_input='residual',
                              sigma_min=0.01, sigma_max=50.0)
    cfg.training = _training('vesde', continuous=True, reduce_mean=False)
    return cfg


def get_ncsnpp_vesde_celeba():
    """NCSN++ discrete VE SDE (SMLD) on CelebA 64x64.
    Reference: score_sde/configs/ve/celeba_ncsnpp.py
    Discrete SMLD: positional (sinusoidal) embedding, sigma_max=90, snr=0.17.
    """
    cfg = ml_collections.ConfigDict()
    cfg.data = ml_collections.ConfigDict()
    cfg.data.dataset = 'celeba'; cfg.data.image_size = 64; cfg.data.num_channels = 3
    cfg.data.centered = False; cfg.data.random_flip = True
    cfg.data.uniform_dequantization = False
    cfg.model = _ncsnpp_model(ema_rate=0.999, embedding_type='positional',
                              fir=True, progressive_input='residual',
                              sigma_min=0.01, sigma_max=90.0)
    cfg.training = _training('vesde', continuous=False, reduce_mean=False)
    cfg.training.snr = 0.17   # CelebA uses 0.17 (vs 0.16 for CIFAR-10)
    return cfg


def get_subvpsde_ddpm():
    """sub-VP SDE, continuous, DDPM model.
    Reference: score_sde/configs/subvp/cifar10_ddpm_continuous.py
    Shares VP beta schedule; only continuous variant exists in reference.
    """
    cfg = ml_collections.ConfigDict()
    cfg.data = _cifar10_data(centered=True)
    cfg.model = _ddpm_model(ema_rate=0.9999, embedding_type='positional',
                            beta_min=0.1, beta_max=20.0)
    cfg.training = _training('subvpsde', continuous=True, reduce_mean=True)
    return cfg


CONFIGS = {
    'vpsde_ddpm_cont':    lambda: get_vpsde_ddpm(continuous=True),
    'vpsde_ddpm_disc':    lambda: get_vpsde_ddpm(continuous=False),
    'vesde_ddpm_disc':    lambda: get_vesde_ddpm(continuous=False),
    'vesde_ddpm_cont':    lambda: get_vesde_ddpm(continuous=True),
    'subvpsde_ddpm_cont': lambda: get_subvpsde_ddpm(),
    'vpsde_ddpmpp_cont':       lambda: get_ddpmpp_vpsde(),
    'vesde_ncsnpp_cont':       lambda: get_ncsnpp_vesde(),
    'vesde_ncsnpp_celeba_disc': lambda: get_ncsnpp_vesde_celeba(),
}


def get_config(name):
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {sorted(CONFIGS)}")
    return CONFIGS[name]()
