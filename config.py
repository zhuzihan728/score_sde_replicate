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

def get_config_local():
    """Tiny config for local testing on CPU/laptop."""
    config = get_config()

    # Shrink everything for fast local testing
    config.training.batch_size = 4
    config.training.n_iters = 100
    config.training.warmup = 10 
    config.model.nf = 32              # 32 channels instead of 128
    config.model.num_res_blocks = 1   # 1 block instead of 4
    config.model.num_scales = 100     # 100 steps instead of 1000

    return config