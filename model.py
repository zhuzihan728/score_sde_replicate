import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import numpy as np
import ml_collections


class GaussianFourierFeatures(nn.Module):
    embed_dim: int = 128
    scale: float = 16.0

    @nn.compact
    def __call__(self, t):
        W = self.param('W', nn.initializers.normal(stddev=self.scale), (self.embed_dim,))
        W = jax.lax.stop_gradient(W)
        t_proj = t[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)


class SinusoidalPosEmb(nn.Module):
    embed_dim: int = 128

    @nn.compact
    def __call__(self, timesteps):
        half_dim = self.embed_dim // 2
        freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / (half_dim - 1))
        args = timesteps[:, None] * freq[None, :]
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class ResnetBlock(nn.Module):
    out_channels: int
    dropout: float = 0.0
    skip_rescale: bool = False

    @nn.compact
    def __call__(self, x, temb, train=False):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=min(C // 4, 32))(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        h = h + nn.Dense(self.out_channels)(nn.swish(temb))[:, None, None, :]
        h = nn.GroupNorm(num_groups=min(self.out_channels // 4, 32))(h)
        h = nn.swish(h)
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        residual = nn.Conv(self.out_channels, kernel_size=(1, 1))(x) if C != self.out_channels else x
        out = h + residual
        if hasattr(self, 'skip_rescale') and self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out


class BigGANResBlock(nn.Module):
    out_channels: int
    dropout: float = 0.0
    skip_rescale: bool = False
    up: bool = False
    down: bool = False
    fir: bool = False

    @nn.compact
    def __call__(self, x, temb, train=False):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=min(C // 4, 32))(x)
        h = nn.swish(h)
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
        h = h + nn.Dense(self.out_channels)(nn.swish(temb))[:, None, None, :]
        h = nn.GroupNorm(num_groups=min(self.out_channels // 4, 32))(h)
        h = nn.swish(h)
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        residual = nn.Conv(self.out_channels, kernel_size=(1, 1))(x) if C != self.out_channels else x
        out = h + residual
        if self.skip_rescale:
            out = out / jnp.sqrt(2.0)
        return out


class AttnBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=min(C // 4, 32))(x)
        q = nn.Conv(C, kernel_size=(1, 1))(h).reshape(B, H * W, C)
        k = nn.Conv(C, kernel_size=(1, 1))(h).reshape(B, H * W, C)
        v = nn.Conv(C, kernel_size=(1, 1))(h).reshape(B, H * W, C)
        attn = nn.softmax(jnp.einsum('bic,bjc->bij', q, k) / jnp.sqrt(C), axis=-1)
        h = nn.Conv(C, kernel_size=(1, 1))(jnp.einsum('bij,bjc->bic', attn, v).reshape(B, H, W, C))
        return h + x


def _fir_kernel():
    kernel_1d = jnp.array([1, 3, 3, 1], dtype=jnp.float32)
    kernel_2d = jnp.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def _upsample_fir(x, kernel):
    B, H, W, C = x.shape
    x_up = jnp.zeros((B, H * 2, W * 2, C), dtype=x.dtype).at[:, ::2, ::2, :].set(x)
    k = jnp.tile(kernel[None, None, :, :], (C, 1, 1, 1))
    x_filtered = jax.lax.conv_general_dilated(
        jnp.transpose(x_up, (0, 3, 1, 2)),
        k * 4.0,
        window_strides=(1, 1),
        padding='SAME',
        feature_group_count=C,
    )
    return jnp.transpose(x_filtered, (0, 2, 3, 1))


def _downsample_fir(x, kernel):
    B, H, W, C = x.shape
    k = jnp.tile(kernel[None, None, :, :], (C, 1, 1, 1))
    x_filtered = jax.lax.conv_general_dilated(
        jnp.transpose(x, (0, 3, 1, 2)),
        k,
        window_strides=(2, 2),
        padding='SAME',
        feature_group_count=C,
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
        return nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)


class Upsample(nn.Module):
    out_channels: int
    fir: bool = False
    method: str = 'nearest'

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = _upsample_fir(x, _fir_kernel()) if self.fir else jax.image.resize(x, (B, H * 2, W * 2, C), method=self.method)
        return nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)


class UNet(nn.Module):
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

        fir              = getattr(config.model, 'fir', False)
        skip_rescale     = getattr(config.model, 'skip_rescale', False)
        resblock_type    = getattr(config.model, 'resblock_type', 'ddpm')
        progressive_input = getattr(config.model, 'progressive_input', 'none')
        upsample_method  = getattr(config.model, 'upsample_method', 'nearest')

        def ResBlock(out_ch, **kwargs):
            if resblock_type == 'biggan':
                return BigGANResBlock(out_channels=out_ch, dropout=dropout,
                                     skip_rescale=skip_rescale, fir=fir, **kwargs)
            return ResnetBlock(out_channels=out_ch, dropout=dropout, skip_rescale=skip_rescale)

        embedding_type = getattr(config.model, 'embedding_type', 'positional')
        if config.training.continuous and embedding_type == 'fourier':
            temb = GaussianFourierFeatures(embed_dim=nf)(time_cond)
        else:
            temb = SinusoidalPosEmb(embed_dim=nf * 2)(time_cond)
        temb = nn.Dense(nf * 4)(temb)
        temb = nn.swish(temb)
        temb = nn.Dense(nf * 4)(temb)

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
                if resblock_type == 'biggan':
                    h = ResBlock(out_ch, down=True)(h, temb, train=train)
                else:
                    h = Downsample(out_ch, fir=fir)(h)
                skips.append(h)
                if progressive_input == 'residual':
                    input_pyramid = _downsample_fir(input_pyramid, _fir_kernel()) if fir else input_pyramid[:, ::2, ::2, :]
                    input_proj = nn.Conv(out_ch, kernel_size=(1, 1))(input_pyramid)
                    h = (h + input_proj) / jnp.sqrt(2.0) if skip_rescale else h + input_proj

        mid_ch = nf * ch_mult[-1]
        h = ResBlock(mid_ch)(h, temb, train=train)
        h = AttnBlock()(h)
        h = ResBlock(mid_ch)(h, temb, train=train)

        for i_level in reversed(range(num_resolutions)):
            out_ch = nf * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                h = ResBlock(out_ch)(jnp.concatenate([h, skips.pop()], axis=-1), temb, train=train)
            if h.shape[1] in attn_resolutions:
                h = AttnBlock()(h)
            if i_level != 0:
                if resblock_type == 'biggan':
                    h = ResBlock(out_ch, up=True)(h, temb, train=train)
                else:
                    h = Upsample(out_ch, fir=fir, method=upsample_method)(h)

        assert not skips
        h = nn.GroupNorm(num_groups=min(nf // 4, 32))(h)
        h = nn.swish(h)
        return nn.Conv(config.data.num_channels, kernel_size=(3, 3), padding='SAME')(h)
