import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # --- Data ---
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'cifar10'
    data.image_size = 32
    data.num_channels = 3
    data.centered = True          # Scale to [-1, 1] instead of [0, 1]
    data.random_flip = True
    data.uniform_dequantization = False  # Only needed for likelihood eval

    # --- Model ---
    config.model = model = ml_collections.ConfigDict()
    model.name = 'ncsnpp'
    model.num_scales = 1000       # N: number of discrete noise levels
    model.sigma_min = 0.01        # VE SDE: smallest noise
    model.sigma_max = 50.0        # VE SDE: largest noise (CIFAR-10)
    model.nf = 128                # Base channel width
    model.ch_mult = (1, 2, 2, 2)  # Channel multipliers at each resolution
    model.num_res_blocks = 4      # ResBlocks per resolution
    model.attn_resolutions = (16,)  # Apply attention at 16x16
    model.dropout = 0.0
    model.ema_rate = 0.999        # EMA for VE models

    # --- Training ---
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 128
    training.n_iters = 1300000    # 1.3M iterations
    training.learning_rate = 2e-4
    training.warmup = 5000
    training.grad_clip = 1.0
    training.continuous = True     # Continuous time (vs discrete)
    training.reduce_mean = False   # VE: sum over pixels, not mean
    training.sde = 'vesde'

    return config

def get_config_local(dataset):
    """Tiny config for local testing on CPU/laptop."""
    config = get_config()

    def get_config_cifar10():
        # Shrink everything for fast local testing
        config.training.batch_size = 4
        config.training.n_iters = 100
        config.training.warmup = 10 
        config.model.nf = 32              # 32 channels instead of 128
        config.model.num_res_blocks = 1   # 1 block instead of 4
        config.model.num_scales = 100     # 100 steps instead of 1000

        return config

    def get_config_celeba():
        """CelebA config with VE SDE."""
        config = get_config()
        config.data.dataset = 'celeb_a'
        config.data.image_size = 64
        config.model.sigma_max = 90.0     # Larger images need more noise
        return config

    def get_config_lsun_bedroom():
        """LSUN Bedroom config with VE SDE."""
        config = get_config()
        config.data.dataset = 'lsun/bedroom'
        config.data.image_size = 256
        config.training.batch_size = 64   # Smaller batch for larger images
        config.model.sigma_max = 348.0    # Even more noise for 256x256
        return config
    
    if dataset in ["cifar10", "celeb_a", "lsun/bedroom"]:
        if dataset == "cifar10":
            return get_config_cifar10()
        elif dataset == "celeb_a":
            return get_config_celeba()
        elif dataset == "lsun/bedroom":
            return get_config_lsun_bedroom()
    else:
        raise ValueError(f"Unknown Dataset: {dataset}")
    