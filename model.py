import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import numpy as np
import ml_collections


class GaussianFourierFeatures(nn.Module):
    """Map scalar time t to a high-dimensional embedding.

    """
    embed_dim: int = 128
    scale: float = 16.0

    @nn.compact
    def __call__(self, t):
        # Random frequencies (fixed, not learned)
        # self.make_rng would also work, but we want deterministic init
        W = self.param('W',
                       nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim,))
        W = jax.lax.stop_gradient(W)

        # Project t onto these frequencies
        # t shape: [B] -> [B, 1] for broadcasting
        t_proj = t[:, None] * W[None, :] * 2 * jnp.pi
        
        # Output both sin and cos -> 2 * embed_dim features
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)

class SinusoidalPosEmb(nn.Module):
    """
    Map an integer timestep to a high-dimensional embedding.

    GaussianFourierFeatures: continuous t in [0,1] → random frequencies
    SinusoidalPosEmb: discrete i in {0,...,N-1} → fixed frequencies
    """
    embed_dim: int = 128

    @nn.compact
    def __call__(self, timesteps):
        # timesteps shape: [B] — integer noise level indices

        half_dim = self.embed_dim // 2
        
        # Frequencies: exponentially spaced from 1 to 1/10000
        # Same formula as "Attention Is All You Need"
        freq = jnp.exp(
            -jnp.log(10000.0) * jnp.arange(half_dim) / (half_dim - 1)
        )
        
        # Outer product: each timestep × each frequency
        # [B, 1] * [half_dim] → [B, half_dim]
        args = timesteps[:, None] * freq[None, :]
        
        # Sin and cos → [B, embed_dim]
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class ResnetBlock(nn.Module):
    """Residual block with time conditioning.
    The "residual" part: output = input + learned_transformation(input, time)
    This makes training easier because the network only needs to learn
    the CHANGE, not the full mapping.
    """
    out_channels: int
    dropout: float = 0.0
    skip_rescale: bool = False

    @nn.compact
    def __call__(self, x, temb, train=False):
        """
        Args:
            x: Image features, shape [B, H, W, C]
            temb: Time embedding, shape [B, time_dim]
            train: Whether we're training (affects dropout)
        """
        B, H, W, C = x.shape

        # --- First conv layer ---
        h = nn.GroupNorm(num_groups=min(C // 4, 32))(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        # --- Inject time information ---
        # Project time embedding to match channel dimension
        temb_proj = nn.Dense(self.out_channels)(nn.swish(temb))
        # Add to feature map: [B, time_dim] -> [B, 1, 1, out_channels]
        h = h + temb_proj[:, None, None, :]

        # --- Second conv layer ---
        h = nn.GroupNorm(num_groups=min(self.out_channels // 4, 32))(h)
        h = nn.swish(h)
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        # --- Residual connection ---
        # If channel count changed, project the skip connection
        if C != self.out_channels:
            residual = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)
        else:
            residual = x

        out = h + residual
        if hasattr(self, 'skip_rescale') and self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out
    
class BigGANResBlock(nn.Module):
    """BigGAN-style residual block with optional up/downsampling inside.
    
    Differences from basic ResnetBlock:
    - Resampling (up/down) happens inside the block, between the two convs
    - Skip connection also gets resampled to match
    """
    out_channels: int
    dropout: float = 0.0
    skip_rescale: bool = False
    up: bool = False
    down: bool = False
    fir: bool = False

    @nn.compact
    def __call__(self, x, temb, train=False):
        B, H, W, C = x.shape

        # First half: norm, activate, resample, conv 
        h = nn.GroupNorm(num_groups=min(C // 4, 32))(x)
        h = nn.swish(h)

        # Resample if needed
        if self.up:
            if self.fir:
                h = _upsample_fir(h, _fir_kernel())
                x = _upsample_fir(x, _fir_kernel())
            else:
                B_, H_, W_, C_ = h.shape
                h = jax.image.resize(h, (B_, H_ * 2, W_ * 2, C_), method='nearest')
                x = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
        elif self.down:
            if self.fir:
                h = _downsample_fir(h, _fir_kernel())
                x = _downsample_fir(x, _fir_kernel())
            else:
                h = h[:, ::2, ::2, :]
                x = x[:, ::2, ::2, :]

        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        # Inject time 
        temb_proj = nn.Dense(self.out_channels)(nn.swish(temb))
        h = h + temb_proj[:, None, None, :]

        # Second half: norm, activate, dropout, conv
        h = nn.GroupNorm(num_groups=min(self.out_channels // 4, 32))(h)
        h = nn.swish(h)
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        # Residual connection 
        if C != self.out_channels:
            residual = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)
        else:
            residual = x

        out = h + residual
        if self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out
    
class AttnBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape

        # Normalize
        h = nn.GroupNorm(num_groups=min(C // 4, 32))(x)

        # Project to queries, keys, values
        q = nn.Conv(C, kernel_size=(1, 1))(h)  # [B, H, W, C]
        k = nn.Conv(C, kernel_size=(1, 1))(h)  # [B, H, W, C]
        v = nn.Conv(C, kernel_size=(1, 1))(h)  # [B, H, W, C]

        # Reshape spatial dims into a sequence: [B, H*W, C]
        q = q.reshape(B, H * W, C)
        k = k.reshape(B, H * W, C)
        v = v.reshape(B, H * W, C)

        attn = jnp.einsum('bic, bjc->bij', q, k) / jnp.sqrt(C)
        attn = nn.softmax(attn, axis=-1)

        h = jnp.einsum('bij,bjc->bic', attn, v)

        # Reshape back to image: [B, H*W, C] -> [B, H, W, C]
        h = h.reshape(B, H, W, C)

        # Project output and add residual
        h = nn.Conv(C, kernel_size=(1, 1))(h)
        return h + x

def _fir_kernel():
    """FIR low-pass filter kernel [1, 3, 3, 1] / 4, following StyleGAN-2."""
    kernel_1d = jnp.array([1, 3, 3, 1], dtype=jnp.float32)
    kernel_2d = jnp.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def _upsample_fir(x, kernel):
    """Upsample by 2x then apply FIR filter."""
    B, H, W, C = x.shape
    x_up = jnp.zeros((B, H * 2, W * 2, C), dtype=x.dtype)
    x_up = x_up.at[:, ::2, ::2, :].set(x)
    # Kernel shape for grouped conv: [out_channels, in_channels/groups, kH, kW]
    # With feature_group_count=C: [C, 1, kH, kW]
    k = jnp.tile(kernel[None, None, :, :], (C, 1, 1, 1))  # [C, 1, kH, kW]
    x_filtered = jax.lax.conv_general_dilated(
        jnp.transpose(x_up, (0, 3, 1, 2)),  # NHWC → NCHW
        k * 4.0,
        window_strides=(1, 1),
        padding='SAME',
        feature_group_count=C
    )
    return jnp.transpose(x_filtered, (0, 2, 3, 1))


def _downsample_fir(x, kernel):
    """Apply FIR filter then downsample by 2x."""
    B, H, W, C = x.shape
    k = jnp.tile(kernel[None, None, :, :], (C, 1, 1, 1))  # [C, 1, kH, kW]
    x_filtered = jax.lax.conv_general_dilated(
        jnp.transpose(x, (0, 3, 1, 2)),
        k,
        window_strides=(2, 2),
        padding='SAME',
        feature_group_count=C
    )
    return jnp.transpose(x_filtered, (0, 2, 3, 1))
class Downsample(nn.Module):
    out_channels: int
    fir: bool = False

    @nn.compact
    def __call__(self, x):
        if self.fir:
            x = _downsample_fir(x, _fir_kernel())
            return nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)
        else:
            return nn.Conv(self.out_channels, kernel_size=(3, 3),
                           strides=(2, 2), padding='SAME')(x)

class Upsample(nn.Module):
    out_channels: int
    fir: bool = False

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        if self.fir:
            x = _upsample_fir(x, _fir_kernel())
        else:
            x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method='nearest')
        return nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)

class UNet(nn.Module):
    """Score model U-Net with configurable architecture improvements."""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=False):
        config = self.config
        nf = config.model.nf
        ch_mult = config.model.ch_mult
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        num_resolutions = len(ch_mult)

        # Architecture improvement flags
        fir = getattr(config.model, 'fir', False)
        skip_rescale = getattr(config.model, 'skip_rescale', False)
        resblock_type = getattr(config.model, 'resblock_type', 'ddpm')
        progressive_input = getattr(config.model, 'progressive_input', 'none')

        # Select residual block type
        def ResBlock(out_ch, **kwargs):
            if resblock_type == 'biggan':
                return BigGANResBlock(out_channels=out_ch, dropout=dropout,
                                     skip_rescale=skip_rescale, fir=fir, **kwargs)
            else:
                return ResnetBlock(out_channels=out_ch, dropout=dropout,
                                  skip_rescale=skip_rescale)

        embedding_type = getattr(config.model, 'embedding_type', 'positional')
        if config.training.continuous and embedding_type == 'fourier':
            temb = GaussianFourierFeatures(embed_dim=nf)(time_cond)
        else:
            temb = SinusoidalPosEmb(embed_dim=nf * 2)(time_cond)
        temb = nn.Dense(nf * 4)(temb)
        temb = nn.swish(temb)
        temb = nn.Dense(nf * 4)(temb)

        # ENCODER
        h = nn.Conv(nf, kernel_size=(3, 3), padding='SAME')(x)
        skips = [h]

        if progressive_input == 'residual':
            input_pyramid = x

        for i_level in range(num_resolutions):
            out_ch = nf * ch_mult[i_level]

            for i_block in range(num_res_blocks):
                h = ResBlock(out_ch)(h, temb, train=train)
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                skips.append(h)

            if i_level != num_resolutions - 1:
                # BigGAN downsamples inside ResBlock; ddpm uses separate module
                if resblock_type == 'biggan':
                    h = ResBlock(out_ch, down=True)(h, temb, train=train)
                else:
                    h = Downsample(out_ch, fir=fir)(h)
                skips.append(h)

                if progressive_input == 'residual':
                    input_pyramid = _downsample_fir(input_pyramid, _fir_kernel()) if fir else input_pyramid[:, ::2, ::2, :]
                    input_proj = nn.Conv(out_ch, kernel_size=(1, 1))(input_pyramid)
                    h = (h + input_proj) / jnp.sqrt(2.0) if skip_rescale else h + input_proj

        # MIDDLE
        mid_ch = nf * ch_mult[-1]
        h = ResBlock(mid_ch)(h, temb, train=train)
        h = AttnBlock()(h)
        h = ResBlock(mid_ch)(h, temb, train=train)

        # DECODER
        for i_level in reversed(range(num_resolutions)):
            out_ch = nf * ch_mult[i_level]

            for i_block in range(num_res_blocks + 1):
                skip = skips.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResBlock(out_ch)(h, temb, train=train)

            # attention once per level, outside the block loop
            if h.shape[1] in attn_resolutions:
                h = AttnBlock()(h)

            if i_level != 0:
                # BigGAN upsamples inside ResBlock; ddpm uses separate module
                if resblock_type == 'biggan':
                    h = ResBlock(out_ch, up=True)(h, temb, train=train)
                else:
                    h = Upsample(out_ch, fir=fir)(h)

        # OUTPUT
        assert not skips
        h = nn.GroupNorm(num_groups=min(nf // 4, 32))(h)
        h = nn.swish(h)
        h = nn.Conv(config.data.num_channels, kernel_size=(3, 3), padding='SAME')(h)

        return h