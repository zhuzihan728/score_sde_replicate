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
        
        # Project t onto these frequencies
        # t shape: [B] -> [B, 1] for broadcasting
        t_proj = t[:, None] * W[None, :] * 2 * jnp.pi
        
        # Output both sin and cos -> 2 * embed_dim features
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)



class ResnetBlock(nn.Module):
    """Residual block with time conditioning.
    The "residual" part: output = input + learned_transformation(input, time)
    This makes training easier because the network only needs to learn
    the CHANGE, not the full mapping.
    """
    out_channels: int
    dropout: float = 0.0

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

        return h + residual
    
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

class Downsample(nn.Module):
    """Reduce spatial resolution by 2x using a strided convolution."""
    out_channels: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.out_channels, kernel_size=(3, 3), 
                       strides=(2, 2), padding='SAME')(x)


class Upsample(nn.Module):
    """Increase spatial resolution by 2x."""
    out_channels: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        # Nearest-neighbor upsampling: double H and W
        x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method='nearest')
        # Then a conv to refine
        return nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)


class UNet(nn.Module):
    """Score model U-Net. Takes (noisy image, time) → score estimate."""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, t, train=False):
        config = self.config
        nf = config.model.nf                    # 128
        ch_mult = config.model.ch_mult          # (1, 2, 2, 2)
        num_res_blocks = config.model.num_res_blocks  # 4 (or 1 for local)
        attn_resolutions = config.model.attn_resolutions  # (16,)
        dropout = config.model.dropout          # 0.0
        num_resolutions = len(ch_mult)          # 4

        # TIME EMBEDDING 
        # t: [B] → temb: [B, nf*4]
        temb = GaussianFourierFeatures(embed_dim=nf)(t)  # [B, nf*2]
        temb = nn.Dense(nf * 4)(temb)                     # [B, nf*4]
        temb = nn.swish(temb)
        temb = nn.Dense(nf * 4)(temb)                     # [B, nf*4]

        # ENCODER
        # Initial conv: [B, 32, 32, 3] → [B, 32, 32, nf]
        h = nn.Conv(nf, kernel_size=(3, 3), padding='SAME')(x)
        skips = [h]  # Store for decoder

        for i_level in range(num_resolutions):
            out_ch = nf * ch_mult[i_level]

            # ResBlocks at this resolution
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_channels=out_ch, dropout=dropout)(
                    h, temb, train=train)
                # Attention at specified resolutions
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                skips.append(h)

            # Downsample (except at the last level)
            if i_level != num_resolutions - 1:
                h = Downsample(out_ch)(h)
                skips.append(h)

        # MIDDLE 
        mid_ch = nf * ch_mult[-1]
        h = ResnetBlock(out_channels=mid_ch, dropout=dropout)(h, temb, train=train)
        h = AttnBlock()(h)
        h = ResnetBlock(out_channels=mid_ch, dropout=dropout)(h, temb, train=train)

        # DECODER 
        for i_level in reversed(range(num_resolutions)):
            out_ch = nf * ch_mult[i_level]

            # ResBlocks at this resolution (one extra for the skip)
            for i_block in range(num_res_blocks + 1):
                # Concatenate skip connection
                skip = skips.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResnetBlock(out_channels=out_ch, dropout=dropout)(
                    h, temb, train=train)
                # Attention at specified resolutions
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)

            # Upsample (except at the last level)
            if i_level != 0:
                h = Upsample(out_ch)(h)

        # OUTPUT 
        assert not skips  # All skips should be consumed
        h = nn.GroupNorm(num_groups=min(nf // 4, 32))(h)
        h = nn.swish(h)
        h = nn.Conv(config.data.num_channels, kernel_size=(3, 3), padding='SAME')(h)

        return h