import ml_collections


# Base Config
def get_config():
    config = ml_collections.ConfigDict()

    # Data 
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'cifar10'
    data.image_size = 32
    data.num_channels = 3
    data.centered = True
    data.random_flip = True
    data.uniform_dequantization = False

    # Model 
    config.model = model = ml_collections.ConfigDict()
    model.name = 'ncsnpp'
    model.num_scales = 1000
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.dropout = 0.0
    model.ema_rate = 0.999
    # VE SDE params
    model.sigma_min = 0.01
    model.sigma_max = 50.0
    # VP SDE params (used when training.sde is vpsde or subvpsde)
    model.beta_min = 0.1
    model.beta_max = 20.0
    
    # Architecture improvements (NCSN++ vs baseline)
    model.fir = True                  # FIR anti-aliased resampling
    model.skip_rescale = True         # Rescale skip connections by 1/√2
    model.resblock_type = 'biggan'    # 'ddpm' (baseline) or 'biggan' (improved)
    model.progressive_input = 'residual'  # 'none' or 'residual' or 'input_skip'
    model.progressive_output = 'none'     # 'none' or 'residual' or 'output_skip'

    # Training 
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 128
    training.n_iters = 1300000
    training.learning_rate = 2e-4
    training.warmup = 5000
    training.grad_clip = 1.0
    training.continuous = True
    training.reduce_mean = False
    # SDE 
    training.sde = 'vesde' # Change to vpsde, subvpsde
    training.sde_N = 1000
    # VESDE settings
    training.sde_sigma_min = 0.01
    training.sde_sigma_max = 50.0
    # VPSDE and Sub-VPSDE settings
    training.sde_beta_min = 0.1
    training.sde_beta_max = 20.0

    # Sampler
    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.denoise = True
    sampler.type = 'PC' # For an ODE sampler, change to ODE 
    # Configurations for PC sampling
    sampler.predictor = 'Euler-Maruyama'
    sampler.corrector = 'Langevin'
    sampler.corrector_snr = 0.1
    sampler.corrector_steps = 5
    sampler.sampler_steps = 5
    # Configurations for ODE sampling
    sampler.rtol=1e-5
    sampler.atol = 1e-5
    sampler.method = 'RK45'
    sampler.eps=1e-3

    return config


# SDE VARIANTS

def get_config_vp():
    config = get_config()
    config.training.sde = 'vpsde'
    config.training.reduce_mean = True
    config.model.ema_rate = 0.9999
    return config

def get_config_subvp():
    config = get_config_vp()
    config.training.sde = 'subvpsde'
    return config

def get_config_ve_discrete():
    """VE SDE with discrete time (baseline NCSN++)."""
    config = get_config()
    config.training.continuous = False
    return config

def get_config_vp_discrete():
    """VP SDE with discrete time (baseline DDPM)."""
    config = get_config_vp()
    config.training.continuous = False
    return config

# DEEP VARIANTS (double residual blocks)

def get_config_ve_deep():
    """VE SDE deep — paper's best FID (2.20)."""
    config = get_config()
    config.model.num_res_blocks = 8
    return config

def get_config_vp_deep():
    """VP SDE deep."""
    config = get_config_vp()
    config.model.num_res_blocks = 8
    return config

def get_config_subvp_deep():
    """Sub-VP SDE deep — paper's best likelihood (2.99 bits/dim)."""
    config = get_config_subvp()
    config.model.num_res_blocks = 8
    return config

# DATASET VARIANTS

def get_config_celeba():
    """VE SDE on CelebA 64x64."""
    config = get_config()
    config.data.dataset = 'celeb_a'
    config.data.image_size = 64
    config.model.sigma_max = 90.0
    config.training.continuous = False
    return config

def get_config_lsun_bedroom():
    """VE SDE on LSUN Bedroom 256x256."""
    config = get_config()
    config.data.dataset = 'lsun/bedroom'
    config.data.image_size = 256
    config.training.batch_size = 64
    config.model.sigma_max = 348.0
    config.model.num_scales = 2000
    return config

# LOCAL TESTING — tiny models for CPU/laptop

def _shrink_for_local(config):
    """Shrink any config for fast local testing."""
    config.training.batch_size = 4
    config.training.n_iters = 100
    config.training.warmup = 10
    config.model.nf = 32
    config.model.num_res_blocks = 1
    config.model.num_scales = 100
    return config

def get_config_local(dataset='cifar10'):
    """Tiny config for local testing. Supports: CIFAR10, celeb_a, lsun/bedroom."""
    if dataset == 'cifar10':
        config = get_config()
    elif dataset == 'celeb_a':
        config = get_config_celeba()
    elif dataset == 'lsun/bedroom':
        config = get_config_lsun_bedroom()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return _shrink_for_local(config)