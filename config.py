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
    data.random_flip = False
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
    training.sde_N = 1_000

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
    config.model.sigma_max = 378
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

def get_config_ddpm_cifar10(continuous=False, pp=False, deep=False, sde_type='vpsde'):
    config = get_config()
    if pp:
        config.model.name = 'ncsnpp'
        config.model.num_res_blocks = 8 if deep else 4
        config.model.fir = False
        config.model.fir_kernel = [1,3,3,1]
        config.model.progressive = 'none'
        config.model.progressive_input = 'none'
        config.model.progressive_combine = 'sum'
        config.model.embedding_type = 'positional'
        config.model.init_scale = 0.
    else:
        config.model.name = 'ddpm'
        config.model.num_res_blocks = 2

    config.model.ema_rate = 0.999
    config.model.nonlinearity = 'swish'
    config.model.normalization = 'GroupNorm'
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.scale_by_sigma = False

    config.training.sde = sde_type
    config.training.continuous = continuous

    config.data.centered = True

    return config

def get_config_ncsnpp_cifar10(continuous=False, deep=False):
    config = get_config()
    config.model.name = 'ncsnpp'
    config.model.num_res_blocks = 8 if deep else 4
    config.model.ema_rate = 0.999
    config.model.nonlinearity = 'swish'
    config.model.normalization = 'GroupNorm'
    config.model.fir = True
    config.model.fir_kernel = [1,3,3,1]
    config.model.progressive = 'none'
    config.model.progressive_input = 'none'
    config.model.progressive_combine = 'sum'
    config.model.embedding_type = 'Fourier' if continuous else 'positional'
    config.model.fourier_scale = 16
    config.model.init_scale = 0.
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.scale_by_sigma = True

    config.training.sde = 'vesde'
    config.training.continuous = continuous

    config.data.centered = True

    return config

def get_config_ncsnpp_bedroom():
    config = get_config_lsun_bedroom()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = True
    config.model.embedding_type = 'fourier'
    config.model.fourier_scale = 16
    config.model.init_scale = 0.
    config.model.ema_rate = 0.999
    config.model.nonlinearity = 'swish'
    config.model.normalization = 'GroupNorm'
    config.model.num_res_blocks = 2
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.fir = True
    config.model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    config.model.fir_kernel = [1,3,3,1]        
    config.model.progressive = 'output_skip'
    config.model.progressive_input = 'input_skip'
    config.model.progressive_combine = 'sum'

    config.training.sde = 'vesde'
    config.training.sde_N = 2_000
    config.training.continous = True
    config.data.centered = False

    return config

def get_config_ncsnpp_celeba():
    config = get_config_ncsnpp_bedroom()
    config.model.sigma_max = 348

    return config

def get_config_ncsnpp_church():
    config = get_config_ncsnpp_bedroom()
    config.model.sigma_max = 380
    return config

def get_config_ncsnpp_ffhq_1024():
    config = get_config_ncsnpp_celeba()
    config.data.image_size = 1024
    config.model.sigma_max = 1348
    config.training.sde_N = 2000
    config.model.ema_rate = 0.9999
    config.model.ch_mult = (1, 2, 4, 8, 16, 32, 32, 32)
    config.model.num_res_blocks = 1
    config.training.batch_size = 8
    config.model.nf = 16

    return config
