import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def rearrange(x, pattern, **kwargs):
    if pattern == 'b c h w -> b (h w) c':
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if pattern == 'b (h w) c -> b c h w':
        b, hw, c = x.shape
        h = kwargs.get('h')
        w = kwargs.get('w')
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    raise NotImplementedError('Pattern not supported in simple rearrange replacement')


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = out.view(b, -1, h, w)
        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type, drop_path_rate=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class BiMFEBlock(nn.Module):
    def __init__(self, dim, dim_prev=None, dim_next=None, num_heads=1, bias=False):
        super(BiMFEBlock, self).__init__()
        self.alpha_down = nn.Parameter(torch.tensor(0.5)) if dim_prev else None
        self.alpha_up = nn.Parameter(torch.tensor(0.5)) if dim_next else None
        if dim_prev:
            self.down_op = nn.Conv2d(dim_prev, dim, kernel_size=3, stride=2, padding=1, bias=bias)
        if dim_next:
            self.up_op = nn.Sequential(
                nn.Conv2d(dim_next, dim * 4, kernel_size=1, bias=bias),
                nn.PixelShuffle(2)
            )
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim, ffn_expansion_factor=2.66, bias=bias)

    def forward(self, x_main, x_prev=None, x_next=None):
        f_fusion = x_main
        if self.alpha_down is not None and x_prev is not None:
            f_fusion = f_fusion + self.alpha_down * self.down_op(x_prev)
        if self.alpha_up is not None and x_next is not None:
            f_fusion = f_fusion + self.alpha_up * self.up_op(x_next)
        x = self.norm1(f_fusion)
        x = self.attn(x)
        x = x_main + x
        x = x + self.ffn(self.norm2(x))
        return x


class GlobalFeatureBranch(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=None,
        num_heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type='WithBias',
        drop_path_rate=0.,
    ):
        super(GlobalFeatureBranch, self).__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        total_blocks = sum(num_blocks) + sum(num_blocks[:-1])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[i])
            for i in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[sum(num_blocks[:1]) + i])
            for i in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[sum(num_blocks[:2]) + i])
            for i in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 4))

        idx = sum(num_blocks[:3])
        self.bimfe_level4 = BiMFEBlock(dim=int(dim * 8), dim_prev=int(dim * 4), dim_next=None, num_heads=num_heads[3], bias=bias)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8), num_heads=num_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[idx + i])
            for i in range(num_blocks[3])
        ])
        idx += num_blocks[3]

        self.bimfe_level3 = BiMFEBlock(dim=int(dim * 4), dim_prev=int(dim * 2), dim_next=int(dim * 8), num_heads=num_heads[2], bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[idx + i])
            for i in range(num_blocks[2])
        ])
        idx += num_blocks[2]

        self.bimfe_level2 = BiMFEBlock(dim=int(dim * 2), dim_prev=int(dim), dim_next=int(dim * 4), num_heads=num_heads[1], bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[idx + i])
            for i in range(num_blocks[1])
        ])
        idx += num_blocks[1]

        self.bimfe_level1 = BiMFEBlock(dim=int(dim), dim_prev=None, dim_next=int(dim * 2), num_heads=num_heads[0], bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=layer_norm_type, drop_path_rate=dpr[idx + i])
            for i in range(num_blocks[0])
        ])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        h, w = inp_img.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        inp_img_padded = F.pad(inp_img, (0, pad_w, 0, pad_h), 'reflect')

        inp_enc_level1 = self.patch_embed(inp_img_padded)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        feat_level4 = self.bimfe_level4(inp_enc_level4, out_enc_level3, None)
        latent_out = self.latent(feat_level4)

        feat_level3 = self.bimfe_level3(out_enc_level3, out_enc_level2, latent_out)
        out_dec_level3 = self.decoder_level3(feat_level3)

        feat_level2 = self.bimfe_level2(out_enc_level2, out_enc_level1, out_dec_level3)
        out_dec_level2 = self.decoder_level2(feat_level2)

        feat_level1 = self.bimfe_level1(out_enc_level1, None, out_dec_level2)
        out_dec_level1 = self.decoder_level1(feat_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img_padded
        return out_dec_level1[:, :, :h, :w]


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(features)))


def window_partition(features: torch.Tensor, window_size: int) -> torch.Tensor:
    batch_size, height, width, channels = features.shape
    features = features.view(batch_size, height // window_size, window_size, width // window_size, window_size, channels)
    return features.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int, batch_size: int) -> torch.Tensor:
    features = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    return features.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, windows: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_windows, token_count, channels = windows.shape
        qkv = self.qkv(windows).view(batch_windows, token_count, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            attention = attention.view(batch_windows // window_count, window_count, self.num_heads, token_count, token_count)
            attention = attention + attention_mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, token_count, token_count)
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).contiguous().view(batch_windows, token_count, channels)
        return self.proj(output)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float = 2.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def _build_attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if self.shift_size == 0:
            return None
        image_mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        width_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        mask_index = 0
        for h_slice in height_slices:
            for w_slice in width_slices:
                image_mask[:, h_slice, w_slice, :] = mask_index
                mask_index += 1
        mask_windows = window_partition(image_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(attention_mask == 0, float(0.0))
        return attention_mask

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        residual = features
        features = features.permute(0, 2, 3, 1).contiguous()
        features = self.norm1(features)
        shifted = torch.roll(features, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else features
        windows = window_partition(shifted, self.window_size)
        attention_mask = self._build_attention_mask(height, width, features.device)
        windows = self.attn(windows, attention_mask=attention_mask)
        shifted = window_reverse(windows, self.window_size, height, width, batch_size)
        features = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted
        features = features.permute(0, 3, 1, 2).contiguous()
        features = residual + self.drop_path(features)
        mlp_input = features.permute(0, 2, 3, 1).contiguous()
        mlp_out = self.mlp(self.norm2(mlp_input)).permute(0, 3, 1, 2).contiguous()
        return features + self.drop_path(mlp_out)


class LocalFeatureBranch(nn.Module):
    def __init__(self, in_channels=3, dim=96, num_blocks=6, num_heads=6, window_size=8, drop_path_rate=0.0):
        super().__init__()
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        dpr = torch.linspace(0, drop_path_rate, steps=num_blocks).tolist()
        blocks = []
        for block_index in range(num_blocks):
            shift_size = 0 if block_index % 2 == 0 else window_size // 2
            blocks.append(SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, mlp_ratio=2.0, drop_path_rate=dpr[block_index]))
        self.body = nn.Sequential(*blocks)
        self.conv_body = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = input_image.shape
        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        padded = F.pad(input_image, (0, pad_width, 0, pad_height), mode='reflect')
        features = self.conv_first(padded)
        body_features = self.body(features)
        body_features = self.conv_body(body_features) + features
        prediction = self.conv_out(body_features) + padded
        return prediction[:, :, :height, :width]


class DualExNet(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        global_dim=48,
        global_num_blocks=None,
        global_num_heads=None,
        global_ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type='WithBias',
        drop_path_rate=0.1,
        local_dim=96,
        local_num_blocks=6,
        local_num_heads=6,
        local_window_size=8,
        self_ensemble=False,
    ):
        super().__init__()
        self.self_ensemble = self_ensemble
        if global_num_blocks is None:
            global_num_blocks = [4, 6, 6, 8]
        if global_num_heads is None:
            global_num_heads = [1, 2, 4, 8]

        self.local_branch = LocalFeatureBranch(
            in_channels=inp_channels,
            dim=local_dim,
            num_blocks=local_num_blocks,
            num_heads=local_num_heads,
            window_size=local_window_size,
            drop_path_rate=drop_path_rate,
        )
        self.global_branch = GlobalFeatureBranch(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=global_dim,
            num_blocks=global_num_blocks,
            num_heads=global_num_heads,
            ffn_expansion_factor=global_ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
            drop_path_rate=drop_path_rate,
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    @staticmethod
    def _x8_transform(tensor: torch.Tensor, mode: int) -> torch.Tensor:
        if mode == 0:
            return tensor
        if mode == 1:
            return tensor.flip(-1)
        if mode == 2:
            return tensor.flip(-2)
        if mode == 3:
            return tensor.flip(-1).flip(-2)
        if mode == 4:
            return tensor.transpose(-2, -1)
        if mode == 5:
            return tensor.transpose(-2, -1).flip(-1)
        if mode == 6:
            return tensor.transpose(-2, -1).flip(-2)
        if mode == 7:
            return tensor.transpose(-2, -1).flip(-1).flip(-2)
        raise ValueError(f'Unsupported transform mode: {mode}')

    def _forward_once(self, input_image: torch.Tensor) -> torch.Tensor:
        local_prediction = self.local_branch(input_image)
        global_prediction = self.global_branch(input_image)
        gate = self.fusion_gate(torch.cat([local_prediction, global_prediction], dim=1))
        return gate * local_prediction + (1.0 - gate) * global_prediction

    def _forward_x8(self, input_image: torch.Tensor) -> torch.Tensor:
        outputs = []
        for mode in range(8):
            aug_input = self._x8_transform(input_image, mode)
            aug_output = self._forward_once(aug_input)
            deaug_output = self._x8_transform(aug_output, mode)
            outputs.append(deaug_output)
        return torch.stack(outputs, dim=0).mean(dim=0)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        if self.self_ensemble and (not self.training):
            return self._forward_x8(input_image)
        return self._forward_once(input_image)
